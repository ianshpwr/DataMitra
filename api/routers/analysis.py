import os
import time
import shutil
import tempfile
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from agents.pipeline import run_static_pipeline, run_live_pipeline
from api.schemas import (
    AnalysisResponse, ErrorResponse,
    InsightOut, EvidenceOut, QualityReportOut, CriticReportOut,
    DecisionOut, DecisionPackOut,   # ✅ ADDED
)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

ALLOWED_EXTENSIONS = {".csv", ".json", ".parquet"}
MAX_FILE_MB = 50


# ── Upload + analyse a static file ───────────────────────────────────────────

@router.post(
    "/analyse/upload",
    response_model=AnalysisResponse,
    summary="Upload a CSV/JSON/Parquet file and get AI insights",
)
async def analyse_upload(
    file:       UploadFile = File(...),
    use_sample: bool       = Query(False, description="Use 10k row sample for speed"),
):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_MB} MB",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = run_static_pipeline(tmp_path, use_sample=use_sample)
        elapsed_ms = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return _build_response(result, elapsed_ms)


# ── Analyse a file already on the server ─────────────────────────────

@router.post(
    "/analyse/path",
    response_model=AnalysisResponse,
    summary="Analyse a file already present on the server",
)
async def analyse_path(
    file_path:  str  = Query(..., description="Absolute or relative path to the file"),
    use_sample: bool = Query(False),
):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        start  = time.time()
        result = run_static_pipeline(file_path, use_sample=use_sample)
        elapsed_ms = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return _build_response(result, elapsed_ms)


# ── Analyse a batch from the live stream ─────────────────────────────

@router.post(
    "/analyse/live",
    response_model=AnalysisResponse,
    summary="Pull a batch from the live Redis stream and analyse it",
)
async def analyse_live():
    try:
        start  = time.time()
        result = run_live_pipeline()
        elapsed_ms = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=503,
            detail=result["error"],
        )

    return _build_response(result, elapsed_ms)


# ── Internal helper ───────────────────────────────────────────────────────────

def _build_response(result: dict, elapsed_ms: int) -> AnalysisResponse:
    bundle = result["final_bundle"]
    ctx    = result["data_ctx"]
    report = result["critic_report"]

    insights_out = [
        InsightOut(
            id=ins.id,
            type=ins.type.value,
            severity=ins.severity.value,
            title=ins.title,
            explanation=ins.explanation,
            evidence=[
                EvidenceOut(metric=e.metric, value=e.value, column=e.column)
                for e in ins.evidence
            ],
            affected_columns=ins.affected_columns,
            confidence=ins.confidence,
            actionable=ins.actionable,
        )
        for ins in bundle.insights
    ]

    # ── ✅ NEW: Decision Pack Mapping ───────────────────────────────────────
    decisions_out = None
    pack = result.get("decision_pack")

    if pack:
        def _dec(d):
            return DecisionOut(
                id=d.id,
                insight_id=d.insight_id,
                action_type=d.action_type.value,
                title=d.title,
                what=d.what,
                why=d.why,
                expected_impact=d.expected_impact,
                impact_level=d.impact_level.value,
                effort_level=d.effort_level.value,
                priority_score=d.priority_score,
                risk_if_ignored=d.risk_if_ignored,
                owner=d.owner,
                kpi=d.kpi,
            )

        decisions_out = DecisionPackOut(
            total_insights=pack.total_insights,
            decisions=[_dec(d) for d in pack.decisions],
            quick_wins=[_dec(d) for d in pack.quick_wins],
            summary=pack.summary,
            generation_ms=pack.generation_ms,
            token_count=pack.token_count,
        )

    return AnalysisResponse(
        success=True,
        domain=bundle.domain,
        source_type=bundle.source_type,
        total_rows=bundle.total_rows,
        executive_summary=bundle.executive_summary,
        insights=insights_out,
        quality=QualityReportOut(
            total_rows=ctx.quality.total_rows,
            duplicate_rows=ctx.quality.duplicate_rows,
            overall_score=ctx.quality.overall_score,
            passed=ctx.quality.passed,
            columns_failing=ctx.quality.columns_failing,
        ),
        critic=CriticReportOut(
            passed_count=report.passed_count,
            rejected_count=report.rejected_count,
            flagged_count=report.flagged_count,
            overall_quality=report.overall_quality,
            needs_human_review=report.needs_human_review,
            retry_was_used=result["retry_count"] > 0,
        ),
        processing_ms=elapsed_ms,
        token_count=bundle.token_count,
        analysed_at=datetime.now(timezone.utc),
        decisions=decisions_out,
        # add to AnalysisResponse call:
        chart_plans=result.get("chart_plans", []),
        df_path=result.get("df_path"),
    )