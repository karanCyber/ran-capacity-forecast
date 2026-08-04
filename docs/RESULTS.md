# Generated results

Evaluation window: `2025-06-10 00:00:00+00:00` to `2025-06-30 23:00:00+00:00` (21 daily forecast origins, 24h horizon, 30,217 scored cell-hours).

## Forecast accuracy

| Model | RMSE | MAE | MASE | Bias | MAPE (util>20%) | MAE gain vs naive |
|---|---|---|---|---|---|---|
| Seasonal naive (same hour last week) | 5.122 | 2.540 | 1.089 | -0.119 | 6.39% | — |
| Seasonal naive + anomaly-cleaned history | 4.342 | 2.311 | 0.991 | -0.237 | 5.78% | 9.0% |
| LightGBM (delta from cleaned baseline) | 3.881 | 1.951 | 0.837 | -0.228 | 4.85% | 23.2% |

## Accuracy by horizon

| Horizon (h) | Baseline MAE | Model MAE | Model MASE | Gain |
|---|---|---|---|---|
| 1 | 1.160 | 0.888 | 0.381 | 23.5% |
| 4 | 0.661 | 0.508 | 0.218 | 23.2% |
| 8 | 3.098 | 2.308 | 0.989 | 25.5% |
| 12 | 3.214 | 2.437 | 1.045 | 24.2% |
| 16 | 3.436 | 2.631 | 1.128 | 23.4% |
| 20 | 3.124 | 2.431 | 1.042 | 22.2% |
| 24 | 1.946 | 1.419 | 0.608 | 27.1% |

> Caveat: every backtest origin is 23:00 UTC, so horizon and hour-of-day are
> perfectly confounded here. The spread across this table is the daily load
> cycle (h=4 is 03:00, quiet; h=8 is 07:00, the morning ramp), **not** error
> growth with lead time. Measuring true lead-time decay needs staggered
> origins; see 'What I would do next'.


## Accuracy by cell archetype

| Cell archetype | Hours scored | Baseline MAE | Model MAE | Gain |
|---|---|---|---|---|
| business | 7052 | 1.959 | 1.483 | 24.3% |
| mixed | 5544 | 2.957 | 2.230 | 24.6% |
| residential | 9062 | 2.974 | 2.227 | 25.1% |
| transit | 8559 | 2.290 | 1.864 | 18.6% |

## Top features by gain

| Feature | Share of total gain |
|---|---|
| week_over_week | 26.1% |
| cell_id | 17.4% |
| lag_168 | 8.4% |
| lag_24 | 6.4% |
| lag_336 | 6.4% |
| lag_72 | 6.1% |
| dayofweek | 4.7% |
| roll_mean_24 | 3.5% |
| lag_48 | 3.4% |
| lag_169 | 3.2% |

## Anomaly detection

| Metric | Value |
|---|---|
| Hourly precision | 0.648 |
| Hourly recall | 0.375 |
| Hourly F1 | 0.475 |
| False positive rate | 0.00666 |
| Hours flagged | 1.83% |
| Alert episodes raised | 146 |
| Episode precision | 0.473 |

### Event-level recall by injected anomaly type

| Injected event type | Events in window | Detected | Recall |
|---|---|---|---|
| drift | 11 | 11 | 1.000 |
| outage | 5 | 5 | 1.000 |
| spike | 14 | 13 | 0.929 |
| ALL | 30 | 29 | 0.967 |

### Threshold sensitivity

| k | Hourly precision | Hourly recall | F1 | Alerts raised | Event recall |
|---|---|---|---|---|---|
| 2.5 | 0.518 | 0.499 | 0.508 | 249 | 1.000 |
| 3.0 | 0.613 | 0.430 | 0.505 | 165 | 0.967 |
| 3.5 | 0.648 | 0.375 | 0.475 | 146 | 0.967 |
| 4.0 | 0.678 | 0.316 | 0.431 | 115 | 0.933 |
| 4.5 | 0.696 | 0.278 | 0.397 | 105 | 0.900 |
| 5.0 | 0.730 | 0.240 | 0.361 | 87 | 0.867 |
