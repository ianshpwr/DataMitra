import streamlit as st
import requests
import time

API = "http://localhost:8000"

st.set_page_config(
    page_title="DataMitra",
    page_icon="📊",
    layout="wide",
)

# ── Minimal style tweaks ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #2d3150;
}
.insight-warning {
    background: #2a1f00;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.insight-critical {
    background: #2a0000;
    border-left: 4px solid #ef4444;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.insight-info {
    background: #1a1f2e;
    border-left: 4px solid #6b7280;
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.exec-box {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.conf-bar { font-family: monospace; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_upload(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        f"{API}/api/v1/analyse/upload",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def call_live() -> dict:
    r = requests.post(f"{API}/api/v1/analyse/live", timeout=60)
    r.raise_for_status()
    return r.json()


def conf_bar(score: float) -> str:
    filled = round(score * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {round(score * 100)}%"


SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


def render_insight(ins: dict):
    sev   = ins["severity"]
    css   = f"insight-{sev}"
    icon  = SEVERITY_ICON.get(sev, "⚪")
    badge = f"[{sev.upper()}]"
    actionable = " ⚡ Action needed" if ins["actionable"] else ""

    st.markdown(f"""
<div class="{css}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
    <span style="font-weight:600;font-size:15px">{icon} {ins['title']}</span>
    <span style="font-size:11px;opacity:0.7">{badge}{actionable}</span>
  </div>
  <p style="margin:0 0 10px;font-size:14px;line-height:1.6;opacity:0.9">{ins['explanation']}</p>
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;opacity:0.6">
    <span>Columns: {', '.join(ins['affected_columns'])}</span>
    <span class="conf-bar">{conf_bar(ins['confidence'])}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_results(data: dict, filename: str):
    # ── Top meta row ─────────────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:13px;color:#6b7280'>Results for <b>{filename}</b></p>",
                unsafe_allow_html=True)

    # ── Quality cards ─────────────────────────────────────────────────────────
    q   = data["quality"]
    pct = round(q["overall_score"] * 100)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Quality score",    f"{pct}%")
    c2.metric("Total rows",       f"{data['total_rows']:,}")
    c3.metric("Duplicates",       f"{q['duplicate_rows']:,}")
    c4.metric("Processing time",  f"{data['processing_ms'] / 1000:.1f}s")
    c5.metric("Tokens used",      f"{data['token_count']:,}")

    st.divider()

    # ── Executive summary ─────────────────────────────────────────────────────
    st.markdown(f"""
<div class="exec-box">
  <p style="font-size:11px;font-weight:600;color:#60a5fa;letter-spacing:0.08em;
            text-transform:uppercase;margin-bottom:10px">Executive Summary</p>
  <p style="font-size:15px;line-height:1.7;color:#e2e8f0;margin:0">
    {data['executive_summary']}
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Two-column layout: insights + critic ──────────────────────────────────
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"#### Insights ({len(data['insights'])})")
        order = {"critical": 0, "warning": 1, "info": 2}
        sorted_ins = sorted(data["insights"],
                            key=lambda x: order.get(x["severity"], 3))
        for ins in sorted_ins:
            render_insight(ins)

    with right:
        # Critic report
        cr = data["critic"]
        st.markdown("#### Critic report")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Passed | ✅ {cr['passed_count']} |
| Rejected | ❌ {cr['rejected_count']} |
| Flagged | 🚩 {cr['flagged_count']} |
| Quality | 📊 {round(cr['overall_quality'] * 100)}% |
| Retried | {'🔄 Yes' if cr['retry_was_used'] else '— No'} |
""")
        if cr["needs_human_review"]:
            st.warning("Human review recommended")

        # Domain + source
        st.markdown("#### Dataset info")
        st.markdown(f"""
| | |
|---|---|
| Domain | `{data['domain']}` |
| Source | `{data['source_type']}` |
| Rows | `{data['total_rows']:,}` |
""")

        # Severity breakdown donut
        counts = {"critical": 0, "warning": 0, "info": 0}
        for ins in data["insights"]:
            counts[ins["severity"]] += 1

        st.markdown("#### Severity breakdown")
        total = sum(counts.values()) or 1
        for sev, cnt in counts.items():
            bar_w = int((cnt / total) * 20)
            st.markdown(
                f"{SEVERITY_ICON[sev]} **{sev.capitalize()}** — {cnt} "
                f"`{'█' * bar_w}`"
            )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("## 📊 DataMitra")
st.markdown(
    "<p style='color:#6b7280;margin-top:-12px'>AI-powered analytics pipeline</p>",
    unsafe_allow_html=True,
)

tab_upload, tab_live = st.tabs(["📁 Upload file", "⚡ Live stream"])

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Drop your CSV, JSON, or Parquet file",
        type=["csv", "json", "parquet"],
    )
    use_sample = st.checkbox("Use 10k row sample (faster)", value=False)

    if uploaded:
        if st.button("Analyse", type="primary", use_container_width=True):
            with st.spinner("Running pipeline — Data Agent → Insight Agent → Critic Agent…"):
                try:
                    data = call_upload(uploaded.getvalue(), uploaded.name)
                    st.session_state["result"]   = data
                    st.session_state["filename"] = uploaded.name
                except requests.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if "result" in st.session_state and st.session_state.get("filename") != "live":
        render_results(st.session_state["result"], st.session_state["filename"])

# ── Live stream tab ───────────────────────────────────────────────────────────
with tab_live:
    st.markdown("Pulls the latest batch from the Redis stream and analyses it.")

    col1, col2 = st.columns([1, 1])
    with col1:
        auto_refresh = st.toggle("Auto-refresh every 30s", value=False)
    with col2:
        if st.button("Pull + analyse now", type="primary", use_container_width=True):
            with st.spinner("Pulling live batch…"):
                try:
                    data = call_live()
                    st.session_state["result"]   = data
                    st.session_state["filename"] = "live"
                    st.session_state["last_live"] = time.time()
                except requests.HTTPError as e:
                    detail = e.response.json().get("detail", e.response.text)
                    st.error(f"{detail}")
                except Exception as e:
                    st.error(str(e))

    if "result" in st.session_state and st.session_state.get("filename") == "live":
        last = st.session_state.get("last_live")
        if last:
            st.caption(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(last))}")
        render_results(st.session_state["result"], "live stream")

    # Auto-refresh logic
    if auto_refresh:
        last = st.session_state.get("last_live", 0)
        if time.time() - last > 30:
            with st.spinner("Auto-refreshing…"):
                try:
                    data = call_live()
                    st.session_state["result"]   = data
                    st.session_state["filename"] = "live"
                    st.session_state["last_live"] = time.time()
                    st.rerun()
                except Exception:
                    pass
        time.sleep(1)
        st.rerun()