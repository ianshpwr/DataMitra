import os, json
from groq import Groq
from dotenv import load_dotenv
import polars as pl

from agents.insight_agent.models import Insight, InsightBundle
from .models import ChartPlan

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

# Patterns that should never appear as chart columns
_SKIP_PATTERNS = [
    "_id", "id", "_key", "_uuid", "_index",
    "lat", "lng", "lon", "latitude", "longitude",
    "zip", "postal", "timestamp", "datetime",
    "created_at", "updated_at",
]

# Change 3 — Rule-based fallback mapping
FALLBACK_RULES: dict[str, tuple[str, str] | None] = {
    "anomaly":                  ("histogram", "count"),
    "distribution":             ("histogram", "count"),
    "categorical_distribution": ("bar",       "count"),
    "trend":                    ("line",      "sum"),
    "time_trend":               ("line",      "sum"),
    "correlation":              ("scatter",   "none"),
    "summary":                  None,   # skip
    "high_nulls":               None,   # skip
    "comparison":               ("bar",  "mean"),
}

# Severity priority for token-budget trimming (Change 5)
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _is_skip_col(col: str) -> bool:
    """Returns True for any column that should never reach the LLM or a chart."""
    col_lower = col.lower()
    return any(pat in col_lower for pat in _SKIP_PATTERNS)


def _make_fallback_plan(ins: Insight) -> ChartPlan | None:
    """
    Change 3 — Rule-based fallback plan, used when LLM is unavailable
    or the insight was not sent to the LLM due to token budget.
    """
    rule = FALLBACK_RULES.get(ins.type)
    if rule is None:
        return None

    chart_type, aggregation = rule

    # Pick x_column: first non-skip affected column
    useful_cols = [c for c in ins.affected_columns if not _is_skip_col(c)]
    if not useful_cols:
        return None

    x_col = useful_cols[0]
    y_col = useful_cols[1] if chart_type == "scatter" and len(useful_cols) > 1 else None

    # Truncate title to 7 words
    words = ins.title.split()
    short_title = " ".join(words[:7]) + ("…" if len(words) > 7 else "")

    return ChartPlan(
        insight_id=ins.id,
        chart_type=chart_type,
        title=short_title,
        subtitle="Auto-generated from insight data",
        x_column=x_col,
        y_column=y_col,
        color_col=None,
        aggregation=aggregation,
        highlight={},
        reasoning="rule-based fallback (LLM unavailable)",
        skip=False,
    )


def _build_slim_stats(ins: Insight, full_stats: dict) -> dict:
    """
    Change 2 — Return only the stats the LLM needs for this insight:
    - Columns in affected_columns
    - OR categorical columns with n_unique <= 15
    Strips nulls, std, n_unique to minimise token usage.
    """
    slim: dict = {}
    affected = set(ins.affected_columns)

    for col, data in full_stats.items():
        include = col in affected
        if not include and data.get("type") == "categorical":
            include = data.get("n_unique", 999) <= 15

        if not include:
            continue

        if data["type"] == "numeric":
            slim[col] = {
                "type": "numeric",
                "mean": data.get("mean"),
                "min":  data.get("min"),
                "max":  data.get("max"),
            }
        elif data["type"] == "categorical":
            slim[col] = {
                "type":     "categorical",
                "top_vals": data.get("top_vals", {}),
            }

    return slim


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


class ChartSelectorAgent:
    """
    Looks at each insight + the actual DataFrame columns/dtypes
    and decides: what chart type, which columns, what to highlight.
    """

    def run(
        self,
        bundle:  InsightBundle,
        df_path: str,
    ) -> list[ChartPlan]:

        # Load real DataFrame
        try:
            df = pl.read_parquet(df_path)
        except Exception as e:
            print(f"  [ChartAgent] Could not load DataFrame: {e}")
            return []

        schema = {
            col: str(dtype)
            for col, dtype in zip(df.columns, df.dtypes)
        }

        # Build full stats dict (unchanged structure, used as source for slim stats)
        full_stats: dict = {}
        for col in df.columns[:15]:  # limit to 15 cols
            if _is_skip_col(col):
                continue
            s = df[col]
            if str(s.dtype) in ("Float64", "Float32", "Int64", "Int32", "Int16", "Int8"):
                full_stats[col] = {
                    "type": "numeric",
                    "mean": round(float(s.mean() or 0), 2),
                    "min":  round(float(s.min() or 0), 2),
                    "max":  round(float(s.max() or 0), 2),
                }
            elif str(s.dtype) == "String":
                vc = s.drop_nulls().value_counts().sort("count", descending=True)
                full_stats[col] = {
                    "type":     "categorical",
                    "n_unique": s.n_unique(),
                    "top_vals": {
                        str(row[col]): int(row["count"])
                        for row in vc.head(5).iter_rows(named=True)
                    },
                }

        def is_useful_column(col: str, series) -> bool:
            col_lower = col.lower()

            # Block IDs
            if col_lower.endswith("_id") or col_lower == "id":
                return False
            if any(k in col_lower for k in ["_index", "uuid", "_key", "hash"]):
                return False

            # Block geo coordinates
            if any(k in col_lower for k in [
                "lat", "lng", "lon", "latitude", "longitude",
                "x_coord", "y_coord"
            ]):
                return False

            # Block raw timestamps
            if any(k in col_lower for k in [
                "created_at", "updated_at", "timestamp",
                "pickup_datetime", "dropoff_datetime"
            ]):
                return False

            # Block high cardinality numeric
            dtype = str(series.dtype)
            if dtype in ("Int64", "Int32", "Int16"):
                unique_ratio = series.n_unique() / max(len(series), 1)
                if unique_ratio > 0.8:
                    return False

            # Block high cardinality string
            if dtype == "String" and series.n_unique() > 50:
                return False

            return True

        insights = bundle.insights

        # ── Change 5 — Token budget: sort by severity, cap at top-5 for LLM ──
        insights_sorted = sorted(
            insights,
            key=lambda i: _SEVERITY_ORDER.get(i.severity, 3)
        )

        # Estimate tokens for a representative prompt
        sample_stats = _build_slim_stats(insights_sorted[0], full_stats) if insights_sorted else {}
        sample_prompt_len = len(json.dumps(sample_stats)) * len(insights_sorted)
        estimated_tokens = _estimate_tokens(str(sample_prompt_len))

        if estimated_tokens > 2000:
            insights_for_llm = insights_sorted[:5]
            insights_for_fallback = insights_sorted[5:]
        else:
            insights_for_llm = insights_sorted
            insights_for_fallback = []

        # ── Change 4 — Pre-batch log ──
        print(
            f"  [ChartAgent] Batch planning {len(insights_for_llm)} insights "
            f"(~{estimated_tokens} tokens)"
        )

        plans: list[ChartPlan] = []
        llm_ok = True
        llm_error: Exception | None = None

        # ── LLM planning loop ──
        for ins in insights_for_llm:
            slim_stats = _build_slim_stats(ins, full_stats)
            plan, from_llm = self._plan_for_insight(ins, schema, slim_stats)
            if plan:
                plans.append(plan)
                # Change 4 — per-plan log
                print(
                    f"  [ChartAgent] {plan.insight_id}: "
                    f"{plan.chart_type} via {'llm' if from_llm else 'fallback'}"
                )
            if not from_llm and llm_ok:
                llm_ok = False

        # ── Fallback for insights that exceeded token budget ──
        for ins in insights_for_fallback:
            plan = _make_fallback_plan(ins)
            if plan:
                plans.append(plan)
                print(
                    f"  [ChartAgent] {plan.insight_id}: "
                    f"{plan.chart_type} via fallback"
                )

        # ── Change 4 — Post-batch log ──
        if llm_ok:
            tokens_used = estimated_tokens   # actual token count not available from Groq SDK easily
            print(
                f"  [ChartAgent] LLM batch succeeded — "
                f"{len(plans)} plans, ~{tokens_used} tokens"
            )
        else:
            print(
                f"  [ChartAgent] Using rule-based fallback for "
                f"{len(insights)} insights (LLM failed: {llm_error})"
            )

        print(f"  [ChartAgent] Generated {len(plans)} chart plans")
        return plans


    def _plan_for_insight(
        self,
        ins:    Insight,
        schema: dict,
        stats:  dict,          # Change 2: slim stats, pre-filtered
    ) -> tuple[ChartPlan | None, bool]:
        """
        Returns (plan, from_llm).
        from_llm=True  → LLM succeeded
        from_llm=False → fallback was used
        """

        prompt = f"""You are a data visualization expert.

You have an insight from a data analysis system and must decide the BEST chart to visualize it.

INSIGHT:
- ID: {ins.id}
- Type: {ins.type}
- Severity: {ins.severity}
- Title: {ins.title}
- Explanation: {ins.explanation}
- Affected columns: {ins.affected_columns}

AVAILABLE COLUMNS AND STATS:
{json.dumps(stats, indent=2)}

RULES:
1. Only use columns that exist in the schema above
2. Choose chart_type from: bar | histogram | line | scatter | pie | heatmap
3. bar: best for categorical comparisons (status counts, region counts, payment methods)
4. histogram: best for showing distribution of a numeric column + outliers
5. line: only if there is a date/time column — shows trend over time
6. scatter: best for showing relationship between 2 numeric columns
7. pie: only for proportions with 2-5 categories
8. heatmap: for correlation between multiple numeric columns
9. If the insight is about null values or data quality only → set skip=true
10. If no chart would add value beyond what text already says → set skip=true

Respond ONLY with valid JSON, no markdown, no explanation:
{{
  "chart_type": "bar",
  "title": "Chart title (max 8 words)",
  "subtitle": "One sentence explaining what the chart reveals",
  "x_column": "column_name or null",
  "y_column": "column_name or null",
  "color_col": "column_name or null",
  "aggregation": "count or mean or sum or none",
  "highlight": {{}},
  "reasoning": "Why this chart type was chosen (1 sentence)",
  "skip": false,
  "skip_reason": ""
}}"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data visualization expert. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw  = response.choices[0].message.content.strip()
            raw  = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            if data.get("skip"):
                return None, True

            # Final guard — reject if chosen column is an ID or geo col
            for col_key in ["x_column", "y_column"]:
                col = data.get(col_key, "") or ""
                if _is_skip_col(col) or col.lower() in {"index", "idx"}:
                    print(f"  [ChartAgent] Rejected non-analytical column '{col}' for {ins.id}")
                    return _make_fallback_plan(ins), False

            return ChartPlan(
                insight_id=ins.id,
                chart_type=data.get("chart_type", "bar"),
                title=data.get("title", ins.title),
                subtitle=data.get("subtitle", ""),
                x_column=data.get("x_column"),
                y_column=data.get("y_column"),
                color_col=data.get("color_col"),
                aggregation=data.get("aggregation", "count"),
                highlight=data.get("highlight", {}),
                reasoning=data.get("reasoning", ""),
                skip=False,
            ), True

        except Exception as e:
            print(f"  [ChartAgent] LLM failed for {ins.id}: {e} — using fallback")
            return _make_fallback_plan(ins), False