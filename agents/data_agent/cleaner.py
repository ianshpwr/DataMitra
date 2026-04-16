import polars as pl
from datetime import datetime
import re


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies all cleaning steps in order.
    Every step is idempotent — safe to run multiple times.
    """
    df = _deduplicate(df)
    df = _normalize_column_names(df)
    df = _fix_date_columns(df)
    df = _fix_currency_columns(df)
    df = _normalize_status_columns(df)
    df = _fill_derived_nulls(df)
    return df


def _deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique()


def _normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """snake_case all column names."""
    rename = {
        col: re.sub(r"\s+", "_", col.strip().lower())
        for col in df.columns
    }
    return df.rename(rename)


def _fix_date_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect columns that look like dates and normalise them to ISO format.
    Handles both YYYY-MM-DD and DD/MM/YYYY.
    """
    date_keywords = ["date", "time", "created", "updated", "at"]
    for col in df.columns:
        if not any(k in col for k in date_keywords):
            continue
        if str(df[col].dtype) != "String":
            continue

        def parse_date(val: str | None) -> str | None:
            if val is None:
                return None
            val = str(val).strip()
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
                        "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y"]:
                try:
                    return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return val  # return as-is if can't parse

        df = df.with_columns(
            pl.col(col).map_elements(parse_date, return_dtype=pl.String)
        )
    return df


def _fix_currency_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Strip $ signs and convert to Float64.
    Targets columns with 'price', 'amount', 'cost', 'value', 'revenue'.
    """
    currency_keywords = ["price", "amount", "cost", "value", "revenue", "salary"]
    for col in df.columns:
        if not any(k in col for k in currency_keywords):
            continue
        if str(df[col].dtype) not in ("String", "Utf8"):
            continue

        df = df.with_columns(
            pl.col(col)
              .str.replace_all(r"[$,£€\s]", "")
              .cast(pl.Float64, strict=False)
              .alias(col)
        )
    return df


def _normalize_status_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase all status/type/category columns."""
    for col in df.columns:
        if any(k in col for k in ["status", "type", "category",
                                   "payment", "risk", "method"]):
            if str(df[col].dtype) == "String":
                df = df.with_columns(
                    pl.col(col).str.to_lowercase().str.strip_chars()
                )
    return df


def _fill_derived_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompute total_amount = quantity * unit_price where it's null.
    Only applies if both source columns exist.
    """
    if all(c in df.columns for c in ["total_amount", "quantity", "unit_price"]):
        df = df.with_columns(
            pl.when(pl.col("total_amount").is_null())
              .then(pl.col("quantity") * pl.col("unit_price"))
              .otherwise(pl.col("total_amount"))
              .alias("total_amount")
        )
    return df