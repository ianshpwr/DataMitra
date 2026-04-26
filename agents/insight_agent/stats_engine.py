import polars as pl
import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass, field


def _is_non_analytical_column(col: str, series) -> bool:
    """
    Returns True if this column should be excluded from analysis.
    Covers: IDs, geo coordinates, timestamps, postal codes.
    """
    col_lower = col.lower()

    # ID columns
    if any(col_lower.endswith(s) for s in ["_id", "id", "_key", "_uuid", "_index"]):
        return True
    if col_lower in {"id", "index", "uuid", "key", "idx"}:
        return True

    # Geo coordinate columns — never useful as distributions
    if any(k in col_lower for k in [
        "lat", "lng", "lon", "latitude", "longitude",
        "pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng",
        "x_coord", "y_coord", "coordinates", "geoloc"
    ]):
        return True

    # Postal/zip codes
    if any(k in col_lower for k in ["zip", "postal", "pincode", "postcode"]):
        return True

    # Raw timestamp columns used as indexes
    if any(k in col_lower for k in [
        "created_at", "updated_at", "timestamp", "datetime",
        "pickup_datetime", "dropoff_datetime", "event_time"
    ]):
        return True

    # High cardinality numeric = likely an ID or coordinate
    if str(series.dtype) in ("Int64", "Int32", "Int16", "Int8"):
        unique_ratio = series.n_unique() / max(len(series), 1)
        if unique_ratio > 0.8:
            return True

    # Float column where ALL values are in coordinate range (-180 to 180)
    # and high cardinality = geo coordinate
    if str(series.dtype) in ("Float64", "Float32"):
        unique_ratio = series.n_unique() / max(len(series), 1)
        if unique_ratio > 0.9:
            try:
                col_min = float(series.min() or 0)
                col_max = float(series.max() or 0)
                if -180 <= col_min and col_max <= 180:
                    return True
            except Exception:
                pass

    return False

@dataclass
class StatResult:
    """One discovered statistical fact."""
    stat_type:   str
    columns:     list[str]
    metrics:     dict
    severity:    str   = "info"   # "info" | "warning" | "critical"
    actionable:  bool  = False


def run_full_analysis(df: pl.DataFrame, domain: str) -> list[StatResult]:
    """
    Entry point. Runs all statistical checks and returns a flat
    list of StatResults, ordered by severity (critical first).
    """
    # Remove non-analytical columns (IDs, coords, timestamps, postal codes)
    useful_cols = [
        c for c in df.columns
        if not _is_non_analytical_column(c, df[c])
    ]
    df = df.select(useful_cols)

    results: list[StatResult] = []
    results += _summary_stats(df)
    results += _null_analysis(df)
    results += _numeric_distributions(df)
    results += _categorical_distributions(df)
    results += _time_trends(df)
    results += _correlations(df)
    results += _anomaly_detection(df)

    order = {"critical": 0, "warning": 1, "info": 2}
    results.sort(key=lambda r: order.get(r.severity, 3))
    return results


# ── 1. Summary stats ──────────────────────────────────────────────────────────
def _summary_stats(df: pl.DataFrame) -> list[StatResult]:
    numeric_cols = [c for c in df.columns
                    if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    if not numeric_cols:
        return []

    metrics = {}
    for col in numeric_cols:
        series = df[col].drop_nulls()
        if len(series) == 0:
            continue
        metrics[col] = {
            "mean":   round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std":    round(float(series.std()), 2),
            "min":    round(float(series.min()), 2),
            "max":    round(float(series.max()), 2),
            "count":  len(series),
        }

    return [StatResult(
        stat_type="summary",
        columns=numeric_cols,
        metrics=metrics,
        severity="info",
        actionable=False,
    )]


# ── 2. Null analysis ──────────────────────────────────────────────────────────
def _null_analysis(df: pl.DataFrame) -> list[StatResult]:
    results = []
    for col in df.columns:
        null_pct = df[col].null_count() / len(df)
        if null_pct > 0.05:
            results.append(StatResult(
                stat_type="high_nulls",
                columns=[col],
                metrics={"null_pct": round(null_pct, 4), "null_count": df[col].null_count()},
                severity="critical" if null_pct > 0.3 else "warning",
                actionable=null_pct > 0.2,
            ))
    return results


# ── 3. Numeric distributions ──────────────────────────────────────────────────
def _numeric_distributions(df: pl.DataFrame) -> list[StatResult]:
    results = []
    numeric_cols = [c for c in df.columns
                    if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    for col in numeric_cols:
        series = df[col].drop_nulls()
        if len(series) < 20:
            continue

        vals      = series.to_numpy()
        mean, std = vals.mean(), vals.std()
        if std == 0:
            continue

        skew     = float(scipy_stats.skew(vals))
        kurt     = float(scipy_stats.kurtosis(vals))
        q1, q3   = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr      = q3 - q1
        outliers = int(((vals < (q1 - 1.5 * iqr)) | (vals > (q3 + 1.5 * iqr))).sum())
        outlier_pct = outliers / len(vals)

        results.append(StatResult(
            stat_type="distribution",
            columns=[col],
            metrics={
                "mean": round(mean, 2),
                "std": round(std, 2),
                "skew": round(skew, 3),
                "kurtosis": round(kurt, 3),
                "outlier_count": outliers,
                "outlier_pct": round(outlier_pct, 4),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
            },
            severity=(
                "critical" if outlier_pct > 0.10
                else "warning" if outlier_pct > 0.03
                else "info"
            ),
            actionable=outlier_pct > 0.05,
        ))
    return results


# ── 4. Categorical distributions ─────────────────────────────────────────────
def _categorical_distributions(df: pl.DataFrame) -> list[StatResult]:
    results = []
    cat_cols = [c for c in df.columns if str(df[c].dtype) == "String"]

    for col in cat_cols:
        series     = df[col].drop_nulls()
        n_unique   = series.n_unique()
        total      = len(series)

        # skip high-cardinality IDs
        if n_unique > 50 or n_unique < 2:
            continue

        counts      = series.value_counts().sort("count", descending=True)
        top_val     = counts[0, col]
        top_count   = int(counts[0, "count"])
        top_pct     = top_count / total
        bottom_pct  = int(counts[-1, "count"]) / total

        # Imbalance: top category dominates
        severity = "info"
        actionable = False
        if top_pct > 0.70:
            severity   = "warning"
            actionable = True
        if top_pct > 0.90:
            severity   = "critical"

        top_5 = {
            str(row[col]): int(row["count"])
            for row in counts.head(5).iter_rows(named=True)
        }

        results.append(StatResult(
            stat_type="categorical_distribution",
            columns=[col],
            metrics={
                "n_unique": n_unique,
                "top_value": top_val,
                "top_pct": round(top_pct, 4),
                "bottom_pct": round(bottom_pct, 4),
                "top_5": top_5,
                "imbalance_ratio": round(top_pct / (bottom_pct + 1e-9), 2),
            },
            severity=severity,
            actionable=actionable,
        ))
    return results


# ── 5. Time-series trends ─────────────────────────────────────────────────────
def _time_trends(df: pl.DataFrame) -> list[StatResult]:
    results   = []
    date_cols = [c for c in df.columns
                 if any(k in c for k in ["date", "time", "created", "at"])
                 and str(df[c].dtype) == "String"]
    num_cols  = [c for c in df.columns
                 if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    if not date_cols or not num_cols:
        return results

    date_col = date_cols[0]

    try:
        df_time = df.with_columns(
            pl.col(date_col)
              .str.slice(0, 10)   # take YYYY-MM-DD part
              .str.to_date(format="%Y-%m-%d", strict=False)
              .alias("_parsed_date")
        ).drop_nulls("_parsed_date")

        if df_time.is_empty():
            return results

        # Monthly aggregation
        df_monthly = (
            df_time
            .with_columns(
                pl.col("_parsed_date").dt.truncate("1mo").alias("_month")
            )
            .group_by("_month")
            .agg([pl.len().alias("order_count")]
                 + [pl.col(c).mean().alias(f"{c}_mean")
                    for c in num_cols[:3]])  # limit to 3 numeric cols
            .sort("_month")
        )

        if len(df_monthly) < 3:
            return results

        # Linear trend on order count
        y = df_monthly["order_count"].to_numpy().astype(float)
        x = np.arange(len(y))
        slope, intercept, r, p_val, _ = scipy_stats.linregress(x, y)

        trend_pct = (slope / (y.mean() + 1e-9)) * 100

        severity   = "info"
        actionable = False
        if abs(trend_pct) > 20:
            severity   = "critical" if trend_pct < -20 else "warning"
            actionable = True
        elif abs(trend_pct) > 10:
            severity = "warning"

        results.append(StatResult(
            stat_type="time_trend",
            columns=[date_col, "order_count"],
            metrics={
                "slope":             round(float(slope), 4),
                "trend_pct_per_period": round(float(trend_pct), 2),
                "r_squared":         round(float(r ** 2), 4),
                "p_value":           round(float(p_val), 4),
                "n_periods":         len(df_monthly),
                "direction":         "up" if slope > 0 else "down",
                "monthly_counts":    df_monthly["order_count"].to_list(),
                "significant":       p_val < 0.05,
            },
            severity=severity,
            actionable=actionable,
        ))
    except Exception as e:
        # Time parsing is fragile — never crash the whole agent on it
        pass

    return results


# ── 6. Correlations ───────────────────────────────────────────────────────────
def _correlations(df: pl.DataFrame) -> list[StatResult]:
    results  = []
    num_cols = [c for c in df.columns
                if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    if len(num_cols) < 2:
        return results

    pairs_checked = 0
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if pairs_checked > 10:  # cap to avoid O(n²) explosion
                break
            ca, cb = num_cols[i], num_cols[j]
            sub    = df.select([ca, cb]).drop_nulls()
            if len(sub) < 30:
                continue

            r, p = scipy_stats.pearsonr(sub[ca].to_numpy(), sub[cb].to_numpy())
            pairs_checked += 1

            if abs(r) > 0.5 and p < 0.05:
                results.append(StatResult(
                    stat_type="correlation",
                    columns=[ca, cb],
                    metrics={
                        "pearson_r": round(float(r), 4),
                        "p_value":   round(float(p), 6),
                        "strength":  "strong" if abs(r) > 0.7 else "moderate",
                        "direction": "positive" if r > 0 else "negative",
                        "n":         len(sub),
                    },
                    severity="info",
                    actionable=False,
                ))
    return results


# ── 7. Anomaly detection (z-score based) ──────────────────────────────────────
def _anomaly_detection(df: pl.DataFrame) -> list[StatResult]:
    results  = []
    num_cols = [c for c in df.columns
                if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    for col in num_cols:
        series = df[col].drop_nulls()
        if len(series) < 30:
            continue

        vals       = series.to_numpy()
        mean, std  = vals.mean(), vals.std()
        if std == 0:
            continue

        z_scores    = np.abs((vals - mean) / std)
        n_extreme   = int((z_scores > 3).sum())  # |z| > 3 = extreme outlier
        extreme_pct = n_extreme / len(vals)

        if n_extreme > 0:
            extreme_vals = vals[z_scores > 3]
            results.append(StatResult(
                stat_type="anomaly",
                columns=[col],
                metrics={
                    "n_anomalies":    n_extreme,
                    "anomaly_pct":    round(extreme_pct, 4),
                    "max_z_score":    round(float(z_scores.max()), 2),
                    "anomaly_values": [round(float(v), 2)
                                       for v in sorted(extreme_vals)[-5:]],
                    "column_mean":    round(float(mean), 2),
                    "column_std":     round(float(std), 2),
                },
                severity=(
                    "critical" if extreme_pct > 0.02
                    else "warning" if n_extreme >= 3
                    else "info"
                ),
                actionable=extreme_pct > 0.01,
            ))
    return results