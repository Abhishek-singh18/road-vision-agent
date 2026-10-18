            m2.metric(
                "Processing FPS",
                stats["fps"] if processor is not None else "—",
            )

            m3, m4 = st.columns(2)
            m3.metric("Top → bottom", stats["top_to_bottom"])
            m4.metric("Bottom → top", stats["bottom_to_top"])

            st.caption(
                f"Processing device: "
                f"{stats.get('device', 'Saved session')}"
            )

        chart_data = pd.DataFrame([
            {
                "Vehicle type": name,
                "Crossings": int(
                    stats["classes"].get(name, 0)
                ),
            }
            for name in ["Car", "Motorcycle", "Bus", "Truck"]
        ])

        with chart_placeholder.container():
            st.markdown("**Vehicle distribution**")

            if int(chart_data["Crossings"].sum()) > 0:
                fig = px.bar(
                    chart_data,
                    x="Vehicle type",
                    y="Crossings",
                    text="Crossings",
                )
                fig.update_layout(
                    height=270,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Crossings",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(
                    fig, use_container_width=True
                )
            else:
                st.caption(
                    "Vehicle distribution will appear after crossings."
                )

        with report_placeholder.container():
            st.markdown("**Crossing event log**")

            if not events.empty:
                st.dataframe(
                    events.tail(20).iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download live CSV",
                    data=events.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="road_vision_live_events.csv",
                    mime="text/csv",
                    key="download_live_events_persistent",
                )
            else:
                st.caption("No crossings recorded yet.")

    render_live_analytics()

st.divider()

reset_col, reset_info = st.columns([1, 3])

with reset_col:
    if st.button(
        "Reset saved live records", type="secondary"
    ):
        processor = webrtc_ctx.video_processor

        if processor is not None:
            processor.reset_stats()
        else:
            clear_saved_events()

        st.success("Saved live records cleared.")
        st.rerun()

with reset_info:
    st.caption(
        "Use this to start a fresh report. Files on hosted services "
        "may not persist across restarts or redeployments."
    )

st.divider()
st.caption(
    "Road Vision Agent · Counts depend on visibility, tracking "
    "stability, camera angle, frame rate, and counting-line placement."
)
