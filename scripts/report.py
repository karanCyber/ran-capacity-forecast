"""Generate the results tables and diagnostic plots from artifacts/metrics.json.

Keeping this scripted means the numbers in the README are regenerated, not
retyped. Retyped numbers drift, and a results table that disagrees with the
artifact is worse than no results table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ran_forecast.config import CONFIG  # noqa: E402

log = logging.getLogger("report")
DOCS = CONFIG.artifact_dir.parent / "docs"


def _md_table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def accuracy_table(metrics: dict) -> str:
    rows = []
    labels = {
        "seasonal_naive": "Seasonal naive (same hour last week)",
        "seasonal_naive_cleaned": "Seasonal naive + anomaly-cleaned history",
        "lightgbm": "LightGBM (delta from cleaned baseline)",
    }
    base_mae = metrics["seasonal_naive"]["mae"]
    for key, label in labels.items():
        if key not in metrics:
            continue
        m = metrics[key]
        rows.append(
            {
                "model": label,
                "rmse": f"{m['rmse']:.3f}",
                "mae": f"{m['mae']:.3f}",
                "mase": f"{m['mase']:.3f}",
                "bias": f"{m['bias']:+.3f}",
                "mape20": f"{m['mape_above_20']:.2f}%",
                "improve": "—" if key == "seasonal_naive"
                else f"{100 * (1 - m['mae'] / base_mae):.1f}%",
            }
        )
    return _md_table(
        rows,
        ["model", "rmse", "mae", "mase", "bias", "mape20", "improve"],
        ["Model", "RMSE", "MAE", "MASE", "Bias", "MAPE (util>20%)", "MAE gain vs naive"],
    )


def anomaly_tables(metrics: dict) -> tuple[str, str, str]:
    anomaly = metrics.get("anomaly", {})
    point = anomaly.get("pointwise", {})
    episodes = anomaly.get("episodes", {})

    summary = _md_table(
        [
            {"metric": "Hourly precision", "value": f"{point.get('precision', 0):.3f}"},
            {"metric": "Hourly recall", "value": f"{point.get('recall', 0):.3f}"},
            {"metric": "Hourly F1", "value": f"{point.get('f1', 0):.3f}"},
            {"metric": "False positive rate", "value": f"{point.get('false_positive_rate', 0):.5f}"},
            {"metric": "Hours flagged", "value": f"{point.get('flagged_pct', 0):.2f}%"},
            {"metric": "Alert episodes raised", "value": episodes.get("episodes", 0)},
            {"metric": "Episode precision", "value": f"{episodes.get('episode_precision', 0):.3f}"},
        ],
        ["metric", "value"],
        ["Metric", "Value"],
    )

    recall = _md_table(
        [
            {
                "kind": r["kind"],
                "events": r["events"],
                "detected": r["detected"],
                "recall": f"{r['recall']:.3f}",
            }
            for r in anomaly.get("event_recall", [])
        ],
        ["kind", "events", "detected", "recall"],
        ["Injected event type", "Events in window", "Detected", "Recall"],
    )

    sweep = _md_table(
        [
            {
                "k": r["k"],
                "precision": f"{r['precision']:.3f}",
                "recall": f"{r['recall']:.3f}",
                "f1": f"{r['f1']:.3f}",
                "episodes": r.get("episodes", ""),
                "event_recall": f"{r.get('event_recall', float('nan')):.3f}",
            }
            for r in anomaly.get("threshold_sweep", [])
        ],
        ["k", "precision", "recall", "f1", "episodes", "event_recall"],
        ["k", "Hourly precision", "Hourly recall", "F1", "Alerts raised", "Event recall"],
    )
    return summary, recall, sweep


def horizon_table(metrics: dict) -> str:
    rows = metrics.get("by_horizon", [])
    keep = [r for r in rows if r["horizon"] in (1, 4, 8, 12, 16, 20, 24)]
    return _md_table(
        [
            {
                "h": r["horizon"],
                "base": f"{r['baseline_mae']:.3f}",
                "model": f"{r['model_mae']:.3f}",
                "mase": f"{r['model_mase']:.3f}",
                "gain": f"{r['improvement_pct']:.1f}%",
            }
            for r in keep
        ],
        ["h", "base", "model", "mase", "gain"],
        ["Horizon (h)", "Baseline MAE", "Model MAE", "Model MASE", "Gain"],
    )


def archetype_table(metrics: dict) -> str:
    return _md_table(
        [
            {
                "a": r["archetype"],
                "n": r["n"],
                "base": f"{r['baseline_mae']:.3f}",
                "model": f"{r['model_mae']:.3f}",
                "gain": f"{r['improvement_pct']:.1f}%",
            }
            for r in metrics.get("by_archetype", [])
        ],
        ["a", "n", "base", "model", "gain"],
        ["Cell archetype", "Hours scored", "Baseline MAE", "Model MAE", "Gain"],
    )


def importance_table(metrics: dict, top: int = 10) -> str:
    rows = metrics.get("feature_importance", [])[:top]
    total = sum(r["gain"] for r in metrics.get("feature_importance", [])) or 1
    return _md_table(
        [
            {"f": r["feature"], "share": f"{100 * r['gain'] / total:.1f}%"}
            for r in rows
        ],
        ["f", "share"],
        ["Feature", "Share of total gain"],
    )


def plot_forecast_example(serving: pd.DataFrame, out: Path) -> None:
    """Actual vs both forecasts for the cell with the most alerts."""
    scored = serving[~serving["is_forecast"]]
    if scored.empty:
        return
    cell_id = scored.groupby("cell_id")["is_anomaly"].sum().idxmax()
    cell = scored[scored["cell_id"] == cell_id].sort_values("timestamp")
    # Centre the window on the cell's most extreme alert rather than simply
    # taking the tail, otherwise the flagged points often fall off the chart.
    flagged_rows = cell[cell["is_anomaly"]]
    if not flagged_rows.empty:
        focus = flagged_rows.loc[flagged_rows["z_score"].abs().idxmax(), "timestamp"]
        lo = focus - pd.Timedelta(days=5)
        hi = focus + pd.Timedelta(days=5)
        cell = cell[(cell["timestamp"] >= lo) & (cell["timestamp"] <= hi)]
    else:
        cell = cell.tail(24 * 10)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(cell["timestamp"], cell["prb_util"], lw=1.6, label="Actual", color="#222222")
    ax.plot(cell["timestamp"], cell["yhat_baseline"], lw=1.0, ls="--",
            label="Seasonal naive", color="#c0392b", alpha=0.8)
    ax.plot(cell["timestamp"], cell["yhat_model"], lw=1.3,
            label="LightGBM", color="#2471a3")

    flagged = cell[cell["is_anomaly"]]
    ax.scatter(flagged["timestamp"], flagged["prb_util"], s=34, zorder=5,
               color="#e67e22", label="Flagged anomaly", edgecolor="white", linewidth=0.5)

    ax.set_title(f"{cell_id} — actual vs forecasts around its largest anomaly")
    ax.set_ylabel("PRB utilisation (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("wrote %s (cell %s)", out, cell_id)


def plot_error_by_hour(serving: pd.DataFrame, out: Path) -> None:
    scored = serving[~serving["is_forecast"]].copy()
    if scored.empty:
        return
    scored["hour"] = scored["timestamp"].dt.hour
    scored["ae_base"] = (scored["prb_util"] - scored["yhat_baseline"]).abs()
    scored["ae_model"] = (scored["prb_util"] - scored["yhat_model"]).abs()
    grouped = scored.groupby("hour")[["ae_base", "ae_model"]].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grouped.index, grouped["ae_base"], marker="o", label="Seasonal naive", color="#c0392b")
    ax.plot(grouped.index, grouped["ae_model"], marker="o", label="LightGBM", color="#2471a3")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean absolute error (PRB %)")
    ax.set_title("Error concentrates at busy hour — which is where capacity decisions are made")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    DOCS.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(CONFIG.metrics_path.read_text())
    serving = pd.read_parquet(CONFIG.forecasts_path)
    serving["timestamp"] = pd.to_datetime(serving["timestamp"], utc=True)

    summary, recall, sweep = anomaly_tables(metrics)
    sections = [
        "# Generated results\n",
        f"Evaluation window: `{metrics['eval_start']}` to `{metrics['eval_end']}` "
        f"({metrics['origins']} daily forecast origins, {metrics['horizon_hours']}h horizon, "
        f"{metrics['lightgbm']['n']:,} scored cell-hours).\n",
        "## Forecast accuracy\n", accuracy_table(metrics), "",
        "## Accuracy by horizon\n", horizon_table(metrics),
        "\n> Caveat: every backtest origin is 23:00 UTC, so horizon and hour-of-day are\n"
        "> perfectly confounded here. The spread across this table is the daily load\n"
        "> cycle (h=4 is 03:00, quiet; h=8 is 07:00, the morning ramp), **not** error\n"
        "> growth with lead time. Measuring true lead-time decay needs staggered\n"
        "> origins; see 'What I would do next'.\n", "",
        "## Accuracy by cell archetype\n", archetype_table(metrics), "",
        "## Top features by gain\n", importance_table(metrics), "",
        "## Anomaly detection\n", summary, "",
        "### Event-level recall by injected anomaly type\n", recall, "",
        "### Threshold sensitivity\n", sweep, "",
    ]
    (DOCS / "RESULTS.md").write_text("\n".join(sections))
    log.info("wrote %s", DOCS / "RESULTS.md")

    plot_forecast_example(serving, DOCS / "forecast_example.png")
    plot_error_by_hour(serving, DOCS / "error_by_hour.png")


if __name__ == "__main__":
    main()
