"""One interface for explicit, offline reference-domain model evaluation.

The live SPS30 pipeline does not call these models: their input domains differ.
Neither score is a diagnosis or authorization to continue/leave an operation.
"""
import argparse
import json
from pathlib import Path


def evaluate(kind, records, domain="unknown"):
    if kind == "pm25":
        from pm25_model import predict_pm25
        prediction = predict_pm25(records, domain=domain)
    elif kind == "physiology":
        from physiology_model import predict_physiology
        prediction = predict_physiology(records, domain=domain)
    else:
        raise ValueError("kind must be pm25 or physiology")
    research_alert = prediction.get("available", False) and prediction.get("research_alert", False)
    return {
        "mode": "offline_research",
        "task": kind,
        "model_result": prediction,
        "health_status": "not_assessed",
        "evacuation_required": None,
        "fit_to_continue": None,
        "review_status": "experimental_alert_review" if research_alert else "no_operational_conclusion",
        "scope": "An experimental label or concentration forecast cannot establish firefighter health or withdrawal necessity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["pm25", "physiology"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON array of reference-domain history records")
    parser.add_argument("--domain", default="unknown", help="Explicit research source assertion, never inferred from field readings")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = evaluate(args.kind, records, args.domain)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
