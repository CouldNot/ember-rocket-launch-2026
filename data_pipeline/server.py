"""Private localhost HTTP API with SQLite event replay on restart.

Run: python server.py --port 8766 --db work/ember-pipeline.sqlite3
The service deliberately listens only on 127.0.0.1. It serves no files and has
no public dashboard, CORS access, publishing endpoint, or remote device gateway.
"""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from pipeline import ExposurePipeline, utc_string


MAX_BODY_BYTES = 1_048_576
MAX_BATCH_RECORDS = 500


def reject_json_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


class PipelineStore:
    def __init__(self, db_path: Path, config=None):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pipeline = ExposurePipeline(config)
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, evaluated_at_utc TEXT NOT NULL, raw_json TEXT NOT NULL, derived_json TEXT NOT NULL)")
        self.connection.commit()
        self.recovered_count = 0
        # The original evaluation clocks reproduce ordering, gaps, and latches.
        for clock, raw in self.connection.execute("SELECT evaluated_at_utc, raw_json FROM observations ORDER BY id"):
            self.pipeline.process(json.loads(raw), now=clock)
            self.recovered_count += 1

    def ingest(self, records, now=None):
        clock = now or datetime.now(timezone.utc)
        with self.lock:
            before = copy.deepcopy(self.pipeline)
            try:
                outputs = []
                with self.connection:
                    for record in records:
                        result = self.pipeline.process(record, now=clock)
                        self.connection.execute(
                            "INSERT INTO observations (evaluated_at_utc, raw_json, derived_json) VALUES (?, ?, ?)",
                            (utc_string(clock) if isinstance(clock, datetime) else clock,
                             json.dumps(record, ensure_ascii=False, allow_nan=False),
                             json.dumps(result, ensure_ascii=False, allow_nan=False)),
                        )
                        outputs.append(result)
                return outputs
            except Exception:
                self.pipeline = before
                raise

    def latest(self, crew, session, now=None):
        with self.lock:
            return self.pipeline.latest(crew, session, now=now)

    def close(self):
        with self.lock:
            self.connection.close()


class ApiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        if address[0] != "127.0.0.1":
            raise ValueError("This private prototype only permits 127.0.0.1")
        self.store = store
        super().__init__(address, ApiHandler)


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "EMBERLocal/1.0"
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def _respond(self, status, data):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _allowed_local_request(self):
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update({"127.0.0.1", "localhost"})
        if self.headers.get("Host", "").lower() not in hosts:
            self._respond(403, {"error": "local_host_required"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin.lower() not in {"http://" + host for host in hosts}:
            self._respond(403, {"error": "cross_origin_request_denied"})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._respond(403, {"error": "cross_site_request_denied"})
            return False
        return True

    def do_GET(self):
        if not self._allowed_local_request():
            return
        url = urlsplit(self.path)
        if url.path == "/health":
            self._respond(200, {
                "service": "ember-observation-api", "status": "ok",
                "listening_scope": "127.0.0.1_only", "private_command_data": True,
                "restart_recovery": "replay_original_events_with_original_evaluation_clocks",
                "recovered_record_count": self.server.store.recovered_count,
                "clinical_model_available": False,
                "clinical_model_unavailable_reason": "current_hardware_missing_hr_core_skin_temperature_and_validated_health_outcomes",
                "evacuation_decision_available": False,
            })
            return
        prefix = "/v1/crew/"
        if url.path.startswith(prefix):
            crew = unquote(url.path[len(prefix):])
            sessions = parse_qs(url.query).get("session_id", [])
            if not crew or len(crew) > 128 or "/" in crew or "\\" in crew or crew in {".", ".."} or len(sessions) != 1 or not sessions[0] or len(sessions[0]) > 128:
                self._respond(400, {"error": "crew_and_single_session_id_required"})
                return
            result = self.server.store.latest(crew, sessions[0])
            if result is None:
                self._respond(404, {"error": "session_not_found"})
            else:
                self._respond(200, result)
            return
        self._respond(404, {"error": "route_not_found"})

    def do_POST(self):
        if not self._allowed_local_request():
            return
        if urlsplit(self.path).path != "/v1/readings":
            self._respond(404, {"error": "route_not_found"})
            return
        if self.headers.get("Transfer-Encoding"):
            self._respond(400, {"error": "chunked_requests_not_supported"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._respond(415, {"error": "application_json_required"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self._respond(411, {"error": "single_content_length_required"})
            return
        try:
            size = int(lengths[0])
        except ValueError:
            self._respond(400, {"error": "invalid_content_length"})
            return
        if not 0 < size <= MAX_BODY_BYTES:
            self._respond(413, {"error": "body_must_be_1_to_1048576_bytes"})
            return
        try:
            body = self.rfile.read(size)
            if len(body) != size:
                self._respond(400, {"error": "incomplete_request_body"})
                return
            payload = json.loads(body.decode("utf-8"), parse_constant=reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TimeoutError, RecursionError):
            self._respond(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._respond(400, {"error": "expected_record_object_or_records_wrapper"})
            return
        is_batch = "records" in payload
        records = payload["records"] if is_batch else [payload]
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_BATCH_RECORDS or any(not isinstance(item, dict) for item in records):
            self._respond(400, {"error": "records_must_be_1_to_500_objects"})
            return
        try:
            outputs = self.server.store.ingest(records)
        except (sqlite3.Error, ValueError, OverflowError):
            self._respond(500, {"error": "local_ingestion_failed_state_rolled_back"})
            return
        if is_batch:
            self._respond(200, {"count": len(outputs), "accepted_interval_count": sum(item["input_accepted"] for item in outputs), "results": outputs})
        else:
            self._respond(200, outputs[0])

    def do_OPTIONS(self):
        self._respond(405, {"error": "cross_origin_access_not_supported"})

    def log_message(self, format, *args):
        # Avoid logging personnel IDs, locations, or session query strings.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--db", type=Path, default=Path("work") / "ember-pipeline.sqlite3")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    store = PipelineStore(args.db, config)
    service = ApiServer(("127.0.0.1", args.port), store)
    print(f"Private local API: http://127.0.0.1:{args.port}; SQLite: {args.db.resolve()}; recovered: {store.recovered_count}", flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()
        store.close()


if __name__ == "__main__":
    main()
