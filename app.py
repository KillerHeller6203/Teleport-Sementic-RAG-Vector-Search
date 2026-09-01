import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(
    page_title="RAG Vector Search Benchmark",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_results():
    path = Path(__file__).parent / "benchmark_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


data = load_results()
eval_metrics = data["evaluation_metrics"]
benchmark_details = data["benchmark_details"]

STRATEGIES = ["A", "B", "HYDE"]
QUERIES = list(eval_metrics.keys())

STRATEGY_COLORS = {
    "A": "#3b82f6",
    "B": "#10b981",
    "HYDE": "#f59e0b",
}

STRATEGY_LABELS = {
    "A": "Strategy A — Baseline",
    "B": "Strategy B — Query Expansion + Reranking",
    "HYDE": "HyDE — Hypothetical Document Embeddings",
}


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


st.markdown(
    """
    <style>
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .winner-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 2.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">🔎 RAG Vector Search Benchmark</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Comparing three retrieval strategies on a distributed systems corpus '
    '— Strategy A (baseline) vs Strategy B (query expansion + reranking) vs HyDE (hypothetical document embeddings)</p>',
    unsafe_allow_html=True,
)

st.markdown("")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        '<div class="stat-card"><div class="stat-value">10</div>'
        '<div class="stat-label">Documents</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="stat-card"><div class="stat-value">49</div>'
        '<div class="stat-label">Chunks</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="stat-card"><div class="stat-value">3</div>'
        '<div class="stat-label">Strategies</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="stat-card"><div class="stat-value">3</div>'
        '<div class="stat-label">Benchmark Queries</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

with st.expander("📐 Pipeline Architecture", expanded=False):
    st.markdown(
        """
        ```
        INGESTION
          Documents → Chunker (256 chars, 32 overlap)
                    → LocalEmbedder (all-MiniLM-L6-v2, 384d)
                    → FAISSVectorStore (IndexFlatIP)

        STRATEGY A (baseline)
          Query → embed → FAISS search → results

        STRATEGY B (enhanced)
          Query → QueryExpander → embed → FAISS
                → MMR rerank → CrossEncoder rerank
                → compress → lost-in-middle reorder → results

        STRATEGY HyDE
          Query → generate 3 hypothetical answer docs
                → embed each → average vectors
                → FAISS → CrossEncoder rerank
                → lost-in-middle reorder → results
        ```
        """
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


st.markdown("## 📊 Metrics Comparison")
st.markdown("")

metric_names = ["mrr", "context_precision", "context_recall"]
metric_labels = {
    "mrr": "MRR (Mean Reciprocal Rank)",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

tab_mrr, tab_prec, tab_recall, tab_latency = st.tabs(
    ["MRR", "Context Precision", "Context Recall", "Latency"]
)

for tab, metric in zip([tab_mrr, tab_prec, tab_recall], metric_names):
    with tab:
        fig = go.Figure()
        for strat in STRATEGIES:
            values = [
                eval_metrics[q].get(strat, {}).get(metric, 0) for q in QUERIES
            ]
            short_queries = [
                q[:35] + "..." if len(q) > 35 else q for q in QUERIES
            ]
            fig.add_trace(
                go.Bar(
                    name=strat,
                    x=short_queries,
                    y=values,
                    marker_color=STRATEGY_COLORS[strat],
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                )
            )
        fig.update_layout(
            title=metric_labels[metric],
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            yaxis=dict(range=[0, 1.15], gridcolor="#334155"),
            xaxis=dict(gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_latency:
    latency_by_strat = {}
    for detail in benchmark_details:
        latency_by_strat.setdefault(detail["strategy"], []).append(
            detail["latency_ms"]
        )

    fig = go.Figure()
    for strat in STRATEGIES:
        lats = latency_by_strat.get(strat, [])
        avg_lat = avg(lats)
        fig.add_trace(
            go.Bar(
                name=strat,
                x=[strat],
                y=[avg_lat],
                marker_color=STRATEGY_COLORS[strat],
                text=[f"{avg_lat:.1f} ms"],
                textposition="outside",
                width=0.4,
            )
        )
    fig.update_layout(
        title="Average Latency per Strategy",
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        yaxis=dict(title="ms", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        showlegend=False,
        height=420,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


st.markdown("## 🏆 Aggregate Leaderboard")
st.markdown("")

agg = {}
for strat in STRATEGIES:
    mrrs = []
    precs = []
    recalls = []
    lats = []
    for q in QUERIES:
        m = eval_metrics[q].get(strat, {})
        mrrs.append(m.get("mrr", 0))
        precs.append(m.get("context_precision", 0))
        recalls.append(m.get("context_recall", 0))
    for d in benchmark_details:
        if d["strategy"] == strat:
            lats.append(d["latency_ms"])

    agg[strat] = {
        "Avg MRR": round(avg(mrrs), 4),
        "Avg Precision": round(avg(precs), 4),
        "Avg Recall": round(avg(recalls), 4),
        "Avg Latency (ms)": round(avg(lats), 1),
    }

leaderboard_df = pd.DataFrame(agg).T
leaderboard_df.index.name = "Strategy"

best_mrr = leaderboard_df["Avg MRR"].idxmax()
best_prec = leaderboard_df["Avg Precision"].idxmax()
best_recall = leaderboard_df["Avg Recall"].idxmax()
fastest = leaderboard_df["Avg Latency (ms)"].idxmin()

lb_col, radar_col = st.columns([1, 1])

with lb_col:
    st.dataframe(
        leaderboard_df.style.highlight_max(
            subset=["Avg MRR", "Avg Precision", "Avg Recall"],
            color="#065f46",
        ).highlight_min(
            subset=["Avg Latency (ms)"],
            color="#065f46",
        ),
        use_container_width=True,
    )

    st.markdown("")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown(f"**🎯 MRR Winner:** `{best_mrr}`")
        st.markdown(f"**📏 Precision Winner:** `{best_prec}`")
    with w2:
        st.markdown(f"**📦 Recall Winner:** `{best_recall}`")
        st.markdown(f"**⚡ Fastest:** `{fastest}`")

with radar_col:
    categories = ["MRR", "Precision", "Recall", "Speed"]
    fig = go.Figure()
    for strat in STRATEGIES:
        vals = agg[strat]
        max_lat = max(v["Avg Latency (ms)"] for v in agg.values())
        speed_norm = 1 - (vals["Avg Latency (ms)"] / max_lat) if max_lat > 0 else 0

        fig.add_trace(
            go.Scatterpolar(
                r=[
                    vals["Avg MRR"],
                    vals["Avg Precision"],
                    vals["Avg Recall"],
                    round(speed_norm, 3),
                ],
                theta=categories,
                fill="toself",
                name=strat,
                line=dict(color=STRATEGY_COLORS[strat]),
                opacity=0.7,
            )
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1], gridcolor="#334155"),
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#334155"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=400,
        margin=dict(t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


st.markdown("## 🔍 Per-Query Breakdown")
st.markdown("")

for query in QUERIES:
    with st.expander(f"**{query}**", expanded=False):
        details_for_query = [
            d for d in benchmark_details if d["query"] == query
        ]
        query_metrics = eval_metrics.get(query, {})

        cols = st.columns(len(details_for_query))

        for col, detail in zip(cols, details_for_query):
            strat = detail["strategy"]
            metrics = query_metrics.get(strat, {})

            with col:
                color = STRATEGY_COLORS[strat]
                st.markdown(
                    f'<span style="color:{color};font-weight:700;font-size:1.1rem">'
                    f"Strategy {strat}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(STRATEGY_LABELS[strat])

                m1, m2 = st.columns(2)
                m1.metric("MRR", f"{metrics.get('mrr', 0):.4f}")
                m2.metric("Hit@3", "✓" if metrics.get("hit_at_k") else "✗")
                m3, m4 = st.columns(2)
                m3.metric("Precision", f"{metrics.get('context_precision', 0):.4f}")
                m4.metric("Recall", f"{metrics.get('context_recall', 0):.4f}")
                st.metric("Latency", f"{detail['latency_ms']:.1f} ms")

                if detail.get("expanded_query"):
                    st.markdown(f"**Expanded query:** _{detail['expanded_query']}_")

                st.markdown("**Retrieved Chunks:**")
                chunk_rows = []
                for r in detail["results"]:
                    chunk_rows.append({
                        "Rank": r["rank"],
                        "Source": r["source_doc_id"],
                        "Score": round(r["score"], 4),
                        "Preview": r["text_preview"][:80] + "…",
                    })
                st.dataframe(
                    pd.DataFrame(chunk_rows),
                    use_container_width=True,
                    hide_index=True,
                )


st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.85rem;">'
    "RAG Vector Search Benchmark — Built with FAISS, Sentence Transformers, and Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
