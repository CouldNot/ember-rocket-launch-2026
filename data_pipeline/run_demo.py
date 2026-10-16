"""Run shipped models and the synthetic sensor replay, without retraining."""
import argparse
import json
from pathlib import Path
from pipeline import ExposurePipeline
from research_inference import evaluate

ROOT = Path(__file__).resolve().parent


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ExposurePipeline(json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
    outputs = []
    for line in (ROOT / "examples" / "pm25_replay.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            result = pipeline.process(record, now=record["received_at_utc"])
            result["evaluation_mode"] = "synthetic_replay"
            outputs.append(result)
    (output_dir / "sensor_replay_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in outputs), encoding="utf-8")
    results = {"live_sensor_replay": {"records": len(outputs), "statuses": sorted({row["status"] for row in outputs}),
               "note": "Synthetic input exercises workflow only; not model validation."}}
    for kind, domain in (("pm25", "uci_beijing_reference"), ("physiology", "prospie_experiment")):
        history = json.loads((ROOT / "data" / f"{kind}_demo_history.json").read_text(encoding="utf-8"))
        results[kind] = evaluate(kind, history, domain)
        assert results[kind]["model_result"]["available"], results[kind]
        manifest_path = ROOT / "reports" / f"{kind}_demo_manifest.json"
        if manifest_path.is_file():
            results[kind]["demonstration_ground_truth"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if kind == "physiology":
            results[kind]["demonstration_note"] = "This held-out positive example was missed by the model. A low model score is not health clearance."
        blocked = evaluate(kind, history, "real_fireground")
        assert not blocked["model_result"]["available"], blocked
        results[kind + "_real_fireground_gate"] = blocked
    assert all(row["evacuation_required"] is None for row in outputs)
    (output_dir / "demo_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Replay and both reference-domain models completed. Outputs: {output_dir}")
    print("Field domain gates verified. No health clearance or evacuation decision produced.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work" / "demo_run")
    run(parser.parse_args().output_dir)
