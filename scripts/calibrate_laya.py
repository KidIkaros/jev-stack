"""Calibration utility for Laya confidence temperatures.

The Laya checkpoint ships with invalid temperatures (outside [0.5, 5]), causing
confidence scores to be uncalibrated. This module fits proper temperatures from
labeled data.

Usage:
    python scripts/calibrate_laya.py --data calibration_data.json

The calibration data format is a list of records:
    [
        {
            "state": "Customer charged $50",
            "questions": {
                "route": {"type": "choice", "criteria": ["billing", "refund", "support"]}
            },
            "targets": {
                "route": [0.0, 1.0, 0.0]  # one-hot: refund is correct
            }
        },
        ...
    ]
"""
import sys
import os
import json
import argparse
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_calibration_data(path: str) -> List[dict]:
    """Load calibration data from JSON file."""
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, list), "Calibration data must be a JSON array"
    return data


def build_pairs(data: List[dict]) -> List[tuple]:
    """Convert calibration data to (state, questions, targets) triples."""
    pairs = []
    for record in data:
        state = record["state"]
        questions = record["questions"]
        targets = record["targets"]
        pairs.append((state, questions, targets))
    return pairs


def run_calibration(agent, data_path: str, save_path: str | None = None):
    """Fit temperatures on labeled data and optionally save the calibration."""
    from laya.calibrate import records_from_labeled, calibration_payload

    data = load_calibration_data(data_path)
    pairs = build_pairs(data)

    print(f"Fitting temperatures on {len(pairs)} labeled records...")
    records = records_from_labeled(agent, pairs)

    print(f"Collected {len(records)} records")
    print(f"QTypes: {set(r[0] for r in records)}")

    result = agent.fit_temperatures(records, compute_ece=True)

    print(f"\nFitted temperatures:")
    print(f"  Type-level: {result['temperature']}")
    print(f"  By options: {result.get('temperature_by_options', {})}")

    if "report" in result:
        report = result["report"]
        print(f"\nECE: {report.get('ece', 'N/A')}")
        print(f"  n: {report.get('n', 'N/A')}")
        print(f"  n_eval: {report.get('n_eval', 'N/A')}")

    if save_path:
        payload = calibration_payload(agent)
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nCalibration saved to {save_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Laya confidence temperatures")
    parser.add_argument("--data", required=True, help="Path to calibration data JSON")
    parser.add_argument("--model", default="ConvaiInnovations/laya", help="Laya model path")
    parser.add_argument("--save", default=None, help="Save calibrated temperatures to file")
    args = parser.parse_args()

    from laya import RLAgent

    agent = RLAgent(model_id_or_path=args.model)
    agent.warmup()

    run_calibration(agent, args.data, args.save)
