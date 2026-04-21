def render_charts(data: dict):
    st.markdown("---")
    st.markdown("## 📈 Visual analytics")

    # ── Row 1: Quality gauge + Severity donut + Action types ─────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(
            chart_quality_gauge(data),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with c2:
        st.plotly_chart(
            chart_severity_donut(data),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with c3:
        fig = chart_action_types(data)
        if fig:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No decisions to chart yet")

    # ── Row 2: Decision priorities ────────────────────────────────────────────
    fig = chart_decision_priorities(data)
    if fig:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Row 3: Numeric distributions ─────────────────────────────────────────
    fig = chart_numeric_distributions(data)
    if fig:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Row 4: Categorical distributions ─────────────────────────────────────
    # Auto-detect which categorical columns exist in the insights
    cat_cols = []
    for ins in data.get("insights", []):
        if ins["type"] == "distribution":
            for ev in ins.get("evidence", []):
                if ev["metric"] == "n_unique":
                    cols = ins.get("affected_columns", [])
                    if cols:
                        cat_cols.append(cols[0])
                    break

    if cat_cols:
        st.markdown("#### Categorical distributions")
        # Up to 3 categorical charts in a row
        chunks = [cat_cols[i:i+3] for i in range(0, len(cat_cols), 3)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, col_name in enumerate(chunk):
                fig = chart_categorical(data, col_name)
                if fig:
                    with cols[i]:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )