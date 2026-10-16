"""Causal PM2.5 observation pipeline; it does not assess fitness for duty.

Only the Python standard library is needed. Measurements are interval averages.
Missing intervals are never filled and ambient PM2.5 is not an inhaled dose.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "expected_interval_seconds": 10.0,
    "stale_after_seconds": 30.0,
    "future_tolerance_seconds": 0.0,
    "minimum_coverage_ratio": 0.8,
    "rolling_window_seconds": 300.0,
    "exposure_alert_threshold_ug_min_m3": None,
    "demo": {
        "enabled": True,
        "elevated_pm25_ug_m3": 150.0,
        "sustain_seconds": 60.0,
        "rise_slope_ug_m3_per_min": 40.0,
        "minimum_trend_points": 3,
        "purpose": "Arbitrary simulation-only parameters; not medical limits",
    },
}
URGENT_CODES = {"request_help", "confusion", "loss_of_consciousness"}
SOURCES = {"sensor", "simulated", "replay"}
QUALITIES = {"valid", "warming_up", "sensor_error", "out_of_range"}
STATUS_LABELS = {
    "monitoring": "持续观测（不代表健康或可继续作业）",
    "check_in": "需要人工核查",
    "urgent_assistance": "紧急人工援助",
    "insufficient_data": "信息不足，需要人工核查",
}


def utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, field: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为带 UTC 时区的 ISO 8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} 时间格式错误") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} 必须明确使用 UTC 时区")
    return parsed.astimezone(timezone.utc)


def finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def safe_json_value(value: Any) -> Any:
    """Keep diagnostics serializable even for direct callers supplying NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json_value(v) for v in value]
    return value


class ExposurePipeline:
    """One sequential processor; the HTTP server protects it with a lock.

    Streams are isolated by firefighter, session, device, and source. Urgent
    reports latch within a firefighter/session/source, across that person's
    devices. A simulated alert cannot contaminate a live sensor stream.
    """

    def __init__(self, config: dict | None = None):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config:
            for key, value in config.items():
                if key == "demo" and isinstance(value, dict):
                    self.config["demo"].update(value)
                elif key in self.config:
                    self.config[key] = value
        self._validate_config()
        self._streams: dict[tuple, dict] = {}
        self._urgent_sessions: dict[tuple, set] = {}
        self._sequence = 0

    def _validate_config(self):
        for name in ("expected_interval_seconds", "stale_after_seconds", "rolling_window_seconds"):
            if not finite_number(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        ratio = self.config["minimum_coverage_ratio"]
        if not finite_number(ratio) or not 0 <= ratio <= 1:
            raise ValueError("minimum_coverage_ratio must be between 0 and 1")
        tolerance = self.config["future_tolerance_seconds"]
        if not finite_number(tolerance) or tolerance < 0:
            raise ValueError("future_tolerance_seconds must be non-negative")
        demo = self.config["demo"]
        if not isinstance(demo["enabled"], bool):
            raise ValueError("demo.enabled must be a boolean")
        for name in ("elevated_pm25_ug_m3", "sustain_seconds", "rise_slope_ug_m3_per_min"):
            if not finite_number(demo[name]) or demo[name] < 0:
                raise ValueError(f"demo.{name} must be a non-negative finite number")
        if not isinstance(demo["minimum_trend_points"], int) or isinstance(demo["minimum_trend_points"], bool) or demo["minimum_trend_points"] < 3:
            raise ValueError("demo.minimum_trend_points must be an integer >= 3")
        # No medical or operational alert is derived from cumulative exposure.
        if self.config["exposure_alert_threshold_ug_min_m3"] is not None:
            raise ValueError("Exposure alerts are disabled: exposure_alert_threshold_ug_min_m3 must be null")

    @staticmethod
    def _now(value: Any = None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, str):
            return parse_utc(value, "now")
        if isinstance(value, datetime) and value.tzinfo is not None:
            return value.astimezone(timezone.utc)
        raise ValueError("now must be an aware datetime or UTC ISO 8601 string")

    @staticmethod
    def _symptoms(record):
        raw = record.get("symptoms", [])
        malformed = not isinstance(raw, list) or any(not isinstance(item, str) for item in raw)
        candidates = raw if isinstance(raw, list) else [raw]
        codes = {item.strip().lower() for item in candidates if isinstance(item, str) and item.strip()}
        return codes, malformed

    def process(self, record: dict, now: Any = None) -> dict:
        at = self._now(now)
        self._sequence += 1
        if not isinstance(record, dict):
            return self._invalid_result({}, at, ["invalid_record"], ["输入必须是 JSON 对象。"])
        codes, malformed_symptoms = self._symptoms(record)
        flags, reasons = [], []
        ids = []
        for field in ("firefighter_id", "session_id", "device_id"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                flags.append("invalid_identity")
                reasons.append(f"{field} 缺失或格式错误；无法归入可靠的数据流。")
            ids.append(value.strip() if isinstance(value, str) else "")
        source = record.get("source")
        if not isinstance(source, str) or source not in SOURCES:
            flags.append("invalid_source")
            reasons.append("source 必须是 sensor、simulated 或 replay。")
        # A broken device identity must never erase a clearly attributed plea.
        if all(isinstance(record.get(field), str) and record[field].strip() and len(record[field]) <= 128
               for field in ("firefighter_id", "session_id")) and isinstance(source, str) and source in SOURCES:
            self._urgent_sessions.setdefault((ids[0], ids[1], source), set()).update(codes & URGENT_CODES)
        if flags:
            result = self._invalid_result(record, at, flags, reasons)
            if codes & URGENT_CODES:
                result.update(status="urgent_assistance", status_label=STATUS_LABELS["urgent_assistance"],
                              health_status="reported_concern", withdrawal_recommendation="urgent_human_assistance")
                result["flags"].append("unattributed_urgent_report")
                result["reasons"].append("收到明确求助，但身份不完整；立即由现场人员介入并核对身份。")
            return result

        key = (*ids, source)
        urgent_key = (ids[0], ids[1], source)
        self._urgent_sessions.setdefault(urgent_key, set()).update(codes & URGENT_CODES)
        if key not in self._streams:
            self._streams[key] = {
                "key": key, "intervals": [], "first_start": None,
                "last_end": None, "last_start": None, "last_valid_end": None,
                "integral_seconds": 0.0, "valid_seconds": 0.0,
                "high_duration": 0.0, "effective_now": at,
                "last_record": {}, "last_flags": [], "last_reasons": [],
                "last_input_accepted": False, "last_accumulated": False,
                "sequence": 0, "current_codes": set(),
            }
        state = self._streams[key]
        if at < state["effective_now"]:
            flags.append("evaluation_clock_regressed")
            reasons.append("评估时间早于已有状态时间，沿用较晚时间以避免回退。")
        state["effective_now"] = max(at, state["effective_now"])
        at = state["effective_now"]
        if malformed_symptoms:
            flags.append("malformed_symptoms")
            reasons.append("symptoms 应为字符串列表；仍独立处理其中可识别的求助信息。")

        start = end = received = None
        for field in ("sample_start_utc", "sample_end_utc", "received_at_utc"):
            try:
                value = parse_utc(record.get(field), field)
                if field == "sample_start_utc": start = value
                elif field == "sample_end_utc": end = value
                else: received = value
            except ValueError as exc:
                flags.append("invalid_timestamp")
                reasons.append(str(exc))
        temporal_ok = start is not None and end is not None and received is not None
        if temporal_ok:
            if end <= start:
                temporal_ok = False
                flags.append("invalid_interval")
                reasons.append("采集结束时间必须晚于开始时间。")
            elif not math.isclose((end - start).total_seconds(), self.config["expected_interval_seconds"], rel_tol=0, abs_tol=0.001):
                flags.append("interval_cadence_mismatch")
                reasons.append("采集平均区间偏离预期聚合时长，需核查设备配置；有效观测仍按实际区间时长积分。")
            if end > at:
                temporal_ok = False
                flags.append("future_sample")
                reasons.append("采集时间位于评估时间之后，未计入任何未来区间。")
            if (received - at).total_seconds() > self.config["future_tolerance_seconds"]:
                temporal_ok = False
                flags.append("future_received_time")
                reasons.append("接收时间位于当前评估时间之后。")
            if received < end:
                temporal_ok = False
                flags.append("received_before_sample_end")
                reasons.append("接收时间早于采集区间结束时间。")
            if state["last_end"] is not None:
                if start == state["last_start"] and end == state["last_end"]:
                    temporal_ok = False
                    flags.append("duplicate_interval")
                    reasons.append("重复采集区间已拒绝，未重复累计。")
                elif start < state["last_end"]:
                    temporal_ok = False
                    flags.append("out_of_order" if end <= state["last_end"] else "overlapping_interval")
                    reasons.append("乱序或重叠区间已拒绝，未重复累计。")
                elif start > state["last_end"]:
                    flags.append("data_gap")
                    reasons.append("采集区间存在缺口；缺口未填补、未按零值处理。")

        quality = record.get("quality_status")
        value = record.get("pm25_ug_m3")
        measurement_ok = True
        if not isinstance(quality, str) or quality not in QUALITIES:
            measurement_ok = False
            flags.append("invalid_quality_status")
            reasons.append("quality_status 缺失或不在允许值内。")
        elif quality != "valid":
            measurement_ok = False
            flags.append("sensor_quality_unusable")
            reasons.append(f"传感器状态为 {quality}；当前区间不计入暴露指数。")
        if value is None:
            measurement_ok = False
            flags.append("missing_pm25")
            reasons.append("PM2.5 缺失，未用零替代。")
        elif not finite_number(value) or value < 0:
            measurement_ok = False
            flags.append("invalid_pm25")
            reasons.append("PM2.5 必须是非负有限数值，布尔值不作为数值。")
        elif temporal_ok:
            numeric_bound = max(600.0, (end - start).total_seconds())
            if not math.isfinite(float(value) * numeric_bound) or not math.isfinite(state["integral_seconds"] + float(value) * (end - start).total_seconds()):
                measurement_ok = False
                flags.append("invalid_pm25")
                reasons.append("PM2.5 数值超出计算可表达范围，未计入积分。")

        # Location and battery do not alter concentration integrals.
        for field, low, high in (("latitude_deg", -90, 90), ("longitude_deg", -180, 180), ("battery_percent", 0, 100)):
            value_aux = record.get(field)
            if value_aux is not None and (not finite_number(value_aux) or not low <= value_aux <= high):
                flags.append("invalid_" + field)
                reasons.append(f"{field} 格式或范围错误，不能用于位置/设备判断。")
        accuracy = record.get("location_accuracy_m")
        if accuracy is not None and (not finite_number(accuracy) or accuracy < 0):
            flags.append("invalid_location_accuracy")
            reasons.append("定位精度必须为非负有限数值。")
        location_time = record.get("location_time_utc")
        if location_time is not None:
            try:
                loc_at = parse_utc(location_time, "location_time_utc")
                if loc_at > at:
                    flags.append("future_location")
                    reasons.append("定位时间位于未来，不能用于当前位置判断。")
            except ValueError as exc:
                flags.append("invalid_location_time")
                reasons.append(str(exc))

        accumulated = False
        if temporal_ok:
            if state["first_start"] is None:
                state["first_start"] = start
            contiguous = state["last_end"] is None or start == state["last_end"]
            state["last_start"], state["last_end"] = start, end
            seconds = (end - start).total_seconds()
            if measurement_ok:
                value = float(value)
                state["intervals"].append((start, end, value))
                state["integral_seconds"] += seconds * value
                state["valid_seconds"] += seconds
                state["last_valid_end"] = end
                accumulated = True
                if value >= self.config["demo"]["elevated_pm25_ug_m3"]:
                    state["high_duration"] = (state["high_duration"] if contiguous else 0) + seconds
                else:
                    state["high_duration"] = 0.0
            else:
                state["high_duration"] = 0.0
        state.update(last_record=safe_json_value(copy.deepcopy(record)), last_flags=flags,
                     last_reasons=reasons, last_input_accepted=temporal_ok,
                     last_accumulated=accumulated, sequence=self._sequence, current_codes=codes)
        return self._derive(state, at)

    def _invalid_result(self, record, at, flags, reasons):
        return {
            "schema_version": "1.0", "firefighter_id": safe_json_value(record.get("firefighter_id")),
            "session_id": safe_json_value(record.get("session_id")), "device_id": safe_json_value(record.get("device_id")),
            "source": safe_json_value(record.get("source")), "evaluated_at_utc": utc_string(at),
            "input_accepted": False, "measurement_accumulated": False,
            "status": "insufficient_data", "status_label": STATUS_LABELS["insufficient_data"],
            "health_status": "not_assessed", "evacuation_required": None,
            "withdrawal_recommendation": "requires_commander_review",
            "observed_exposure_index_ug_min_m3": None, "valid_duration_seconds": 0,
            "observed_time_weighted_pm25_ug_m3": None, "coverage_ratio": None,
            "rolling_5min_mean": None, "trend": {"direction": "insufficient_data", "slope_ug_m3_per_min": None},
            "latest_reading_age": None, "flags": list(dict.fromkeys(flags)), "reasons": reasons,
            "evaluation_mode": "live", "urgent_latched": False,
            "clinical_model_available": False,
            "clinical_model_unavailable_reason": "current_hardware_missing_hr_core_skin_temperature_and_validated_health_outcomes",
        }

    def _derive(self, state, at):
        from datetime import timedelta
        at = max(at, state["effective_now"])
        flags, reasons = list(state["last_flags"]), list(state["last_reasons"])
        duration = state["valid_seconds"]
        elapsed = max(0.0, (at - state["first_start"]).total_seconds()) if state["first_start"] else 0.0
        coverage = min(1.0, duration / elapsed) if elapsed > 0 else None
        age = (at - state["last_valid_end"]).total_seconds() if state["last_valid_end"] else None
        start_window = at - timedelta(seconds=self.config["rolling_window_seconds"])
        roll_total = roll_seconds = 0.0
        points = []
        for begin, end, val in state["intervals"]:
            if end <= start_window or begin >= at:
                continue
            overlap = max(0.0, (min(at, end) - max(start_window, begin)).total_seconds())
            roll_total += overlap * val
            roll_seconds += overlap
            if end <= at:
                points.append(((end - at).total_seconds() / 60.0, val))
        slope = None
        if len(points) >= 3:
            avg_x = sum(x for x, _ in points) / len(points)
            avg_y = sum(y for _, y in points) / len(points)
            denom = sum((x - avg_x) ** 2 for x, _ in points)
            if denom > 0:
                slope = sum((x - avg_x) * (y - avg_y) for x, y in points) / denom
        direction = "insufficient_data" if slope is None else ("rising" if slope > 1e-9 else "falling" if slope < -1e-9 else "stable")
        if age is None or age > self.config["stale_after_seconds"]:
            flags.append("no_valid_measurement" if age is None else "stale_measurement")
            reasons.append("当前没有可用的新鲜有效读数，需要人工核查。")
        if coverage is not None and coverage < self.config["minimum_coverage_ratio"]:
            flags.append("low_coverage")
            reasons.append("有效观测覆盖不足；累计值仅代表实际观测到的区间。")
        urgent_codes = sorted(self._urgent_sessions.get((state["key"][0], state["key"][1], state["key"][3]), set()))
        status = "monitoring"
        critical = {
            "invalid_timestamp", "invalid_interval", "future_sample", "future_received_time",
            "received_before_sample_end", "duplicate_interval", "out_of_order", "overlapping_interval",
            "data_gap", "invalid_quality_status", "sensor_quality_unusable", "missing_pm25", "invalid_pm25",
            "no_valid_measurement", "stale_measurement", "low_coverage", "malformed_symptoms",
            "interval_cadence_mismatch",
        }
        if set(flags) & critical:
            status = "insufficient_data"
        elif flags or state["current_codes"]:
            status = "check_in"
        demo_active = state["key"][3] == "simulated" and self.config["demo"]["enabled"]
        if demo_active and status == "monitoring":
            demo = self.config["demo"]
            if state["high_duration"] >= demo["sustain_seconds"] and state["high_duration"] > 0:
                status = "check_in"
                flags.append("simulation_elevated_signal")
                reasons.append("模拟浓度达到演示参数并持续；仅用于展示人工核查流程，不是医学阈值。")
            if slope is not None and len(points) >= demo["minimum_trend_points"] and slope >= demo["rise_slope_ug_m3_per_min"]:
                status = "check_in"
                flags.append("simulation_rising_signal")
                reasons.append("模拟浓度上升趋势达到演示参数；仅用于演示，不判定健康或撤离。")
        if state["current_codes"]:
            reasons.append("收到人员主动报告：" + ", ".join(sorted(state["current_codes"])) + "。")
        if urgent_codes:
            status = "urgent_assistance"
            flags.append("urgent_report_latched")
            reasons.append("本次作业已收到明确求助或严重症状报告，需立即由现场人员介入；本原型不会自动解除求助。")
        if not reasons:
            reasons.append("当前有效数据用于持续观测；尚未评估人员健康或适岗情况。")
        last = state["last_record"]
        return {
            "schema_version": "1.0", "firefighter_id": state["key"][0], "session_id": state["key"][1],
            "device_id": state["key"][2], "source": state["key"][3],
            "evaluated_at_utc": utc_string(at), "evaluation_mode": "live",
            "input_accepted": state["last_input_accepted"], "measurement_accumulated": state["last_accumulated"],
            "status": status, "status_label": STATUS_LABELS[status],
            "health_status": "reported_concern" if urgent_codes or state["current_codes"] else "not_assessed",
            "evacuation_required": None,
            "withdrawal_recommendation": "urgent_human_assistance" if urgent_codes else "requires_commander_review",
            "urgent_latched": bool(urgent_codes), "urgent_report_codes": urgent_codes,
            "observed_exposure_index_ug_min_m3": state["integral_seconds"] / 60.0,
            "valid_duration_seconds": duration,
            "observed_time_weighted_pm25_ug_m3": state["integral_seconds"] / duration if duration else None,
            "session_elapsed_seconds": elapsed, "coverage_ratio": coverage,
            "coverage_scope": "since_first_observed_interval_start; unobserved earlier shift time unknown",
            "rolling_5min_mean": roll_total / roll_seconds if roll_seconds else None,
            "rolling_observed_seconds": roll_seconds,
            "rolling_window_seconds": self.config["rolling_window_seconds"],
            "trend": {"direction": direction, "slope_ug_m3_per_min": slope, "point_count": len(points)},
            "latest_reading_age": age, "latest_reading_age_unit": "seconds_since_last_valid_sample_end",
            "last_valid_sample_end_utc": utc_string(state["last_valid_end"]) if state["last_valid_end"] else None,
            "latest_received_at_utc": last.get("received_at_utc"),
            "latest_pm25_ug_m3": last.get("pm25_ug_m3") if state["last_accumulated"] else None,
            "location": {field: last.get(field) for field in ("latitude_deg", "longitude_deg", "location_time_utc", "location_accuracy_m")},
            "battery_percent": last.get("battery_percent"),
            "flags": list(dict.fromkeys(flags)), "reasons": reasons,
            "simulation_rules_active": demo_active, "clinical_model_available": False,
            "clinical_model_unavailable_reason": "current_hardware_missing_hr_core_skin_temperature_and_validated_health_outcomes",
            "limitations": "环境 PM2.5 观测指数不是吸入剂量；不能识别 CO、缺氧或个体热应激，也不能决定是否继续作业。",
        }

    def latest(self, crew: str, session: str, now: Any = None) -> dict | None:
        """Return one isolated stream, preferring live sensors, never a sum.

        The endpoint includes available stream identities when multiple devices
        or sources exist. Re-evaluation at query time reveals subsequent staleness.
        """
        matches = [state for key, state in self._streams.items() if key[0] == crew and key[1] == session]
        if not matches:
            return None
        source_priority = {"sensor": 2, "replay": 1, "simulated": 0}
        state = max(matches, key=lambda item: (source_priority[item["key"][3]], item["sequence"]))
        at = self._now(now)
        result = self._derive(state, at)
        result["available_streams"] = []
        for item in matches:
            view = self._derive(item, at)
            result["available_streams"].append({field: view[field] for field in ("device_id", "source", "status", "urgent_latched", "health_status")})
        result["stream_selection"] = "sensor_before_replay_before_simulated_then_most_recent; streams_are_never_merged"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="UTF-8 JSONL derived output")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    pipeline = ExposurePipeline(config)
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8-sig") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON number: " + value)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise SystemExit(f"Invalid JSON at line {line_number}: {exc}") from exc
            clock_value = record.get("received_at_utc") or record.get("sample_end_utc") if isinstance(record, dict) else None
            try:
                replay_now = parse_utc(clock_value, "replay_clock")
            except ValueError:
                replay_now = datetime.now(timezone.utc)
            result = pipeline.process(record, now=replay_now)
            result["evaluation_mode"] = "replay"
            dst.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    print(f"Processed {count} records in replay evaluation mode -> {args.output}")


if __name__ == "__main__":
    main()
