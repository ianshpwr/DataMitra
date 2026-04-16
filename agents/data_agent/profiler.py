import polars as pl
import duckdb
from .models import ColumnProfile

# Heuristics to tag columns with semantic meaning
SEMANTIC_RULES = {
    "currency":  lambda name, vals: (
        any(k in name.lower() for k in ["price", "amount", "cost", "revenue", "value", "salary"])
        or any(str(v).startswith("$") for v in vals[:20] if v is not None)
    ),
    "date":      lambda name, vals: (
        any(k in name.lower() for k in ["date", "time", "at", "created", "updated"])
    ),
    "id":        lambda name, vals: (
        name.lower().endswith("_id") or name.lower() == "id"
    ),
    "status":    lambda name, vals: (
        "status" in name.lower() and isinstance(vals[0] if vals else None, str)
    ),
    "category":  lambda name, vals: (
        any(k in name.lower() for k in ["category", "type", "kind", "group", "segment"])
    ),
    "email":     lambda name, vals: (
        "email" in name.lower()
    ),
    "geo":       lambda name, vals: (
        any(k in name.lower() for k in ["city", "country", "region", "state", "zip"])
    ),
}


def detect_semantic_tag(name: str, sample_values: list) -> str | None:
    for tag, rule in SEMANTIC_RULES.items():
        try:
            if rule(name, sample_values):
                return tag
        except Exception:
            pass
    return None


def detect_column_issues(col: pl.Series) -> list[str]:
    """Detect specific data quality problems for a single column."""
    issues = []
    dtype  = str(col.dtype)
    name   = col.name

    if col.null_count() / len(col) > 0.15:
        issues.append(f"high_null_rate_{col.null_count() / len(col):.0%}")

    if dtype == "String":
        sample = col.drop_nulls().head(200).to_list()

        # Mixed date formats
        if any(k in name.lower() for k in ["date", "time"]):
            has_iso   = any("/" not in str(v) and "-" in str(v) for v in sample)
            has_slash = any("/" in str(v) for v in sample)
            if has_iso and has_slash:
                issues.append("mixed_date_formats")

        # Currency strings
        if any(str(v).startswith("$") for v in sample):
            issues.append("leading_dollar_sign")

        # Case inconsistency (for low-cardinality columns)
        unique_vals = col.drop_nulls().unique().to_list()
        if 2 <= len(unique_vals) <= 20:
            lower_set = {str(v).lower() for v in unique_vals}
            if len(lower_set) < len(unique_vals):
                issues.append("case_inconsistency")

    return issues


def profile_dataframe(df: pl.DataFrame) -> list[ColumnProfile]:
    """Generate a ColumnProfile for every column in the DataFrame."""
    profiles = []
    for col_name in df.columns:
        col         = df[col_name]
        sample_vals = col.drop_nulls().head(5).to_list()
        issues      = detect_column_issues(col)
        tag         = detect_semantic_tag(col_name, sample_vals)

        profiles.append(ColumnProfile(
            name=col_name,
            dtype=str(col.dtype),
            null_pct=round(col.null_count() / len(df), 4),
            unique_count=col.n_unique(),
            sample_vals=sample_vals,
            semantic_tag=tag,
            issues=issues,
        ))
    return profiles