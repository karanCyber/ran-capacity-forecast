"""One-command end-to-end check.

Runs the test suite, confirms every artifact exists and is internally
consistent, boots the API in-process and exercises each endpoint, then prints
the headline numbers. Intended for a reviewer who has just cloned the repo and
wants to know in one command whether the claims in the README hold.

    make verify
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ran_forecast.config import CONFIG

PASS = "  PASS"
FAIL = "  FAIL"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL}  {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def run_tests() -> None:
    section("1. Test suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "no output"
    check("pytest", result.returncode == 0, tail)


def check_artifacts() -> dict:
    section("2. Artifacts")
    required = {
        "hourly dataset": CONFIG.hourly_path,
        "trained model": CONFIG.model_path,
        "model metadata": CONFIG.model_meta_path,
        "serving forecasts": CONFIG.forecasts_path,
        "anomaly episodes": CONFIG.artifact_dir / "episodes.parquet",
        "metrics": CONFIG.metrics_path,
    }
    for label, path in required.items():
        check(label, path.exists(), str(path.name) if path.exists() else f"missing: {path}")

    if not CONFIG.metrics_path.exists():
        print("\nArtifacts missing. Run `make train` first.")
        sys.exit(1)
    return json.loads(CONFIG.metrics_path.read_text())


def check_consistency(metrics: dict) -> None:
    section("3. Internal consistency")
    serving = pd.read_parquet(CONFIG.forecasts_path)
    serving["timestamp"] = pd.to_datetime(serving["timestamp"], utc=True)

    check(
        "forecasts stay within [0, 100]",
        bool(serving["yhat_model"].between(0, 100).all()),
        f"min {serving['yhat_model'].min():.2f}, max {serving['yhat_model'].max():.2f}",
    )

    future = serving[serving["is_forecast"]]
    check(
        "forward forecast covers every cell for the full horizon",
        len(future) == serving["cell_id"].nunique() * CONFIG.horizon_hours,
        f"{len(future)} rows for {serving['cell_id'].nunique()} cells",
    )
    check(
        "future rows have no actuals (no leakage into the forecast window)",
        bool(future["prb_util"].isna().all()),
    )

    scored = serving[~serving["is_forecast"]]
    check(
        "scored rows all have actuals",
        bool(scored["prb_util"].notna().all()),
    )

    base = metrics["seasonal_naive"]
    model = metrics["lightgbm"]
    check(
        "model beats the seasonal-naive baseline on MAE",
        model["mae"] < base["mae"],
        f"{model['mae']:.3f} vs {base['mae']:.3f}",
    )
    check(
        "model beats the seasonal-naive baseline on RMSE",
        model["rmse"] < base["rmse"],
        f"{model['rmse']:.3f} vs {base['rmse']:.3f}",
    )
    check(
        "MASE below 1.0 (i.e. genuinely better than same-hour-last-week)",
        model["mase"] < 1.0,
        f"MASE {model['mase']:.3f}",
    )
    check(
        "both models scored on identical rows",
        base["n"] == model["n"],
        f"{model['n']:,} cell-hours",
    )

    anomaly = metrics.get("anomaly", {})
    events = anomaly.get("event_recall", [])
    overall = next((e for e in events if e["kind"] == "ALL"), None)
    if overall:
        check(
            "event-level recall above 0.90",
            overall["recall"] >= 0.90,
            f"{overall['detected']}/{overall['events']} events detected",
        )


def check_api() -> None:
    section("4. API endpoints")
    from fastapi.testclient import TestClient

    from ran_forecast.api.main import app

    with TestClient(app) as client:
        check("GET /healthz", client.get("/healthz").status_code == 200)

        ready = client.get("/readyz")
        check("GET /readyz", ready.status_code == 200,
              f"{ready.json().get('cells')} cells loaded" if ready.status_code == 200 else "")

        cells = client.get("/cells")
        check("GET /cells", cells.status_code == 200)
        cell_id = cells.json()["cells"][0]["cell_id"]

        forecast = client.get(f"/forecast/{cell_id}?horizon=24")
        ok = forecast.status_code == 200
        check(f"GET /forecast/{cell_id}", ok)
        if ok:
            body = forecast.json()
            check(
                "forecast returns the full horizon",
                len(body["points"]) == 24,
                f"{len(body['points'])} points",
            )
            check(
                "baseline returned alongside the model forecast",
                all(p["yhat_baseline"] is not None for p in body["points"]),
            )

        check("GET /forecast/UNKNOWN returns 404",
              client.get("/forecast/DOES_NOT_EXIST").status_code == 404)

        anomalies = client.get("/anomalies?limit=5")
        check("GET /anomalies", anomalies.status_code == 200,
              f"{anomalies.json()['count']} episodes returned")

        risk = client.get("/capacity-risk?limit=5")
        check("GET /capacity-risk", risk.status_code == 200,
              f"{len(risk.json())} cells at risk")

        check("GET /metrics-summary", client.get("/metrics-summary").status_code == 200)


def print_headline(metrics: dict) -> None:
    section("5. Headline results")
    rows = [
        ("Seasonal naive", metrics["seasonal_naive"]),
        ("Naive + cleaned history", metrics.get("seasonal_naive_cleaned")),
        ("LightGBM", metrics["lightgbm"]),
    ]
    print(f"  {'Model':<26}{'RMSE':>8}{'MAE':>8}{'MASE':>8}")
    for label, m in rows:
        if m:
            print(f"  {label:<26}{m['rmse']:>8.3f}{m['mae']:>8.3f}{m['mase']:>8.3f}")
    print(f"\n  MAE improvement over baseline: {metrics['mae_improvement_pct']:.1f}%")

    anomaly = metrics.get("anomaly", {})
    point = anomaly.get("pointwise", {})
    overall = next((e for e in anomaly.get("event_recall", []) if e["kind"] == "ALL"), None)
    if point:
        print(f"  Anomaly hourly precision:      {point['precision']:.3f}")
    if overall:
        print(f"  Anomaly event recall:          {overall['recall']:.3f} "
              f"({overall['detected']}/{overall['events']})")


def main() -> int:
    print("ran-capacity-forecast — end-to-end verification")
    run_tests()
    metrics = check_artifacts()
    check_consistency(metrics)
    check_api()
    print_headline(metrics)

    section("Result")
    if _failures:
        print(f"  {len(_failures)} check(s) failed:")
        for item in _failures:
            print(f"    - {item}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
