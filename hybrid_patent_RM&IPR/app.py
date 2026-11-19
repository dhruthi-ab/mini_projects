"""Hybrid AI Patent Prior-Art Retrieval & Analysis Platform.

Features:
- Okapi BM25 Lexical Matching with patent domain stopwords
- Dense Semantic Search with Sentence Transformers & caching
- Two-Stage Cross-Encoder Neural Re-Ranking
- Novelty Risk Gauge & Keyword Differential Breakdown
- 2D Semantic Patent Landscape Visualizer (Plotly)
- Evaluation Benchmark Studio (Precision@K, Recall@K, nDCG@K, MAP, Alpha Sweep)
- Model Fine-Tuning Studio (Train on patent pairs/triplets)
- Dataset Manager (Real USPTO patents & custom portfolio uploader)
"""

import json
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from search_engine import HybridPatentSearch
from evaluator import evaluate_engine_benchmarks
from trainer import PatentModelTrainer
from download_uspto import download_and_format

# Configure Page
st.set_page_config(
    page_title="Hybrid Patent AI | Prior-Art Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for an Executive, Modern Dashboard
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .patent-card {
        background: #1e293b;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        padding: 18px 24px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .keyword-badge-pos {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.82rem;
    }
    .keyword-badge-novel {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.82rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Preset Inventions for Instant Demo Testing
PRESET_INVENTIONS = {
    "Select a pre-configured invention...": {
        "title": "",
        "abstract": "",
        "claims": "",
        "category": "All Domains"
    },
    "AI: Sparse Multi-Head Attention Transformer": {
        "title": "Dynamic Routing Sparse Transformer for Long Sequence Attention",
        "abstract": "A neural sequence model partitions query-key attention into factorized memory blocks using dynamic gating. Memory complexity scales linearly O(N) rather than quadratically O(N^2) while retaining cross-context representations.",
        "claims": "1. An attention processor computing factorized token projections; a gating routing module routing query tokens to sparse memory blocks; and a context synthesizer outputting transformed sequence representations.",
        "category": "G - Physics & Computing"
    },
    "Energy: Solid-State Lithium Battery with Silicon Anode": {
        "title": "Solid-State Lithium Battery with Sulfide Ceramic Electrolyte",
        "abstract": "A high-energy-density rechargeable electrochemical cell combines a lithium phosphorus sulfide solid electrolyte with a silicon nanowire anode and an elastic polymer buffer to suppress lithium dendrite formation.",
        "claims": "1. An electrochemical battery comprising: an anode comprising silicon nanowires; a sulfide solid-state ceramic electrolyte layer; and an elastic fluoroelastomer interfacial buffer suppressing mechanical stress.",
        "category": "H - Electricity"
    },
    "Robotics: Autonomous Vehicle LiDAR-Radar-Camera Fusion": {
        "title": "Multi-Modal Bird's-Eye-View Sensor Fusion for Autonomous Navigation",
        "abstract": "An autonomous driving perception system integrates 3D point clouds from solid-state LiDAR, Doppler velocity radar, and multi-camera RGB feeds into a unified bird's-eye-view temporal feature map.",
        "claims": "1. A vehicle perception unit comprising: a solid-state LiDAR array; radar sensors; cameras; and a deep neural cross-attention network fusing spatial sensor streams into bird's-eye-view trajectory predictions.",
        "category": "B - Transporting & Operations"
    },
    "MedTech: Non-Invasive Optical Raman Glucose Monitor": {
        "title": "Wearable Raman Spectroscopy Sensor for Transcutaneous Glucose Detection",
        "abstract": "A non-invasive continuous glucose monitor projects multi-wavelength near-infrared laser pulses into epidermal tissue, measuring backscattered Raman emission spectra with machine learning noise filtering.",
        "claims": "1. A diagnostic device comprising: a multi-wavelength laser diode; an optical spectrometer collecting backscattered Raman spectra from capillary beds; and a regression model isolating blood glucose peaks.",
        "category": "A - Human Necessities"
    },
    "IoT: Smart Hydration Bottle with Capacitive Level Sensor": {
        "title": "Connected Drinking Vessel with Automated Hydration Telemetry",
        "abstract": "A smart insulated water bottle measures real-time fluid consumption using longitudinal capacitive sensing strips and transmits scheduled hydration alerts to a connected smartphone application.",
        "claims": "1. A hydration tracking container comprising: an insulated fluid reservoir; capacitive strip sensors detecting liquid volume levels; a Bluetooth transceiver; and a microcontroller calculating daily hydration targets.",
        "category": "A - Human Necessities"
    }
}


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Hybrid Patent AI")
st.sidebar.caption("Enterprise Prior-Art Retrieval & Novelty Analytics")

# 1. Dataset Selection
st.sidebar.subheader("📂 Document Corpus")
corpus_options = {
    "Curated Benchmark Corpus (220 Patents)": "data/patents_real_benchmark.csv",
    "Large USPTO Real Dataset (1,000 Patents)": "data/uspto_patents_large.csv",
    "Original Demo Dataset (20 Synthetic)": "data/patents_demo.csv"
}

# Check if fine-tuned model checkpoint exists
finetuned_path = "models/patent_hybrid_finetuned"
has_finetuned = os.path.exists(finetuned_path) and os.path.exists(os.path.join(finetuned_path, "model.safetensors"))

corpus_choice = st.sidebar.selectbox("Active Patent Corpus", list(corpus_options.keys()))
csv_path = corpus_options[corpus_choice]

# If file doesn't exist, create it on demand
if not os.path.exists(csv_path):
    if "uspto_patents_large" in csv_path:
        with st.sidebar.status("Downloading USPTO patent dataset..."):
            download_and_format(1000, csv_path)
    else:
        csv_path = "data/patents_demo.csv"

# 2. Embedding Model Selection
st.sidebar.subheader("🧠 Semantic Neural Model")
model_options = {
    "MiniLM-L6-v2 (Ultra Fast)": "sentence-transformers/all-MiniLM-L6-v2",
    "PatentSBERTa (Domain-Adapted)": "AI-Growth-Lab/PatentSBERTa"
}
if has_finetuned:
    model_options["🎯 Fine-Tuned Patent Model (Local Checkpoint)"] = finetuned_path

model_choice = st.sidebar.selectbox("Embedding Model", list(model_options.keys()))
model_name = model_options[model_choice]

# 3. Retrieval Pipeline Mode
st.sidebar.subheader("⚙️ Retrieval Architecture")
retrieval_mode = st.sidebar.selectbox(
    "Scoring Pipeline",
    [
        ("Hybrid Calibrated Score Fusion", "hybrid_weighted"),
        ("Two-Stage Neural Re-Ranking (Cross-Encoder)", "neural_rerank"),
        ("Reciprocal Rank Fusion (RRF)", "hybrid_rrf"),
        ("Pure Semantic Dense Search", "semantic_only"),
        ("Pure Lexical BM25 Search", "lexical_only")
    ],
    format_func=lambda x: x[0]
)[1]

lexical_weight = 0.5
if retrieval_mode in ["hybrid_weighted", "neural_rerank"]:
    lexical_weight = st.sidebar.slider(
        "Lexical Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="α weights BM25 lexical overlap. 1 - α weights dense semantic similarity."
    )

top_k = st.sidebar.slider("Number of Top Documents", min_value=1, max_value=20, value=5)


# Search Engine Caching
@st.cache_resource(show_spinner=False)
def load_engine(path: str, m_name: str):
    return HybridPatentSearch(csv_path=path, model_name=m_name)


with st.spinner("Initializing Hybrid Patent Index (Okapi BM25 + Dense Embeddings)..."):
    try:
        search_engine = load_engine(csv_path, model_name)
    except Exception as exc:
        st.error(f"Error loading search engine: {exc}")
        st.stop()

# Sidebar Corpus Stats
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Corpus Size:** `{len(search_engine.patents):,}` patents")
st.sidebar.markdown(f"**Embedding Dim:** `{search_engine.embeddings.shape[1]}` dimensions")
st.sidebar.markdown(f"**Cache Status:** `Active & Vector-Indexed`")


# Header Section
col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("⚖️ Hybrid AI Patent Retrieval Platform")
    st.markdown(
        "Advanced Prior-Art Search combining **Okapi BM25 Term Saliency**, **Dense Semantic Encoders**, "
        "and **Cross-Encoder Neural Re-Ranking** with live novelty analytics."
    )
with col_status:
    st.markdown("""
    <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 12px; text-align: center;'>
        <span style='color: #10b981; font-weight: 700; font-size: 1.1rem;'>● Online</span><br/>
        <span style='font-size: 0.85rem; color: #94a3b8;'>Hybrid Engine Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# Main Dashboard Tabs
tabs = st.tabs([
    "🔎 Prior-Art Search & Analyzer",
    "🌐 2D Semantic Patent Map",
    "📊 Benchmark & Evaluation Studio",
    "🚀 Model Fine-Tuning Studio",
    "📁 Dataset Explorer & Ingestion"
])


# ==============================================================================
# TAB 1: PRIOR-ART SEARCH & INVENTION ANALYZER
# ==============================================================================
with tabs[0]:
    st.subheader("Describe Your Technical Invention")
    
    preset_selection = st.selectbox(
        "Load a sample patent disclosure:",
        list(PRESET_INVENTIONS.keys())
    )
    selected_preset = PRESET_INVENTIONS[preset_selection]

    with st.form("search_form"):
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            title_input = st.text_input(
                "Invention Title",
                value=selected_preset["title"],
                placeholder="e.g. Dynamic Routing Sparse Transformer for Long Sequence Attention"
            )
        with col_t2:
            domain_filter = st.selectbox(
                "IPC Technology Filter",
                ["All Domains", "G - Physics & Computing", "H - Electricity", "A - Human Necessities", "B - Transporting & Operations", "C - Chemistry & Metallurgy", "F - Mechanical Engineering"],
                index=0
            )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            desc_input = st.text_area(
                "Technical Abstract / Detailed Description",
                value=selected_preset["abstract"],
                placeholder="Describe structural features, mechanical arrangement, algorithmic operations, or chemical formulations...",
                height=130
            )
        with col_d2:
            claims_input = st.text_area(
                "Key Independent Claim (Optional, boosts claim alignment)",
                value=selected_preset["claims"],
                placeholder="1. A system comprising: a first processing module configured to...",
                height=130
            )

        submitted = st.form_submit_button("🔍 Search Prior Art & Assess Novelty", type="primary", use_container_width=True)

    if submitted or title_input.strip():
        if not title_input.strip() and not desc_input.strip():
            st.warning("Please provide at least an invention title or description to search.")
        else:
            full_query = f"{title_input.strip()}. {desc_input.strip()}"
            if claims_input.strip():
                full_query += f". Claims: {claims_input.strip()}"

            with st.spinner("Executing multi-stage retrieval & novelty risk scoring..."):
                start_t = time.time()
                results_df, novelty = search_engine.search(
                    query=full_query,
                    top_k=top_k,
                    lexical_weight=lexical_weight,
                    mode=retrieval_mode,
                    ipc_filter=domain_filter if domain_filter != "All Domains" else None
                )
                search_time_ms = round((time.time() - start_t) * 1000, 1)

            st.markdown("### 📋 Executive Prior-Art Assessment")

            # Top Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Retrieved Documents", f"{len(results_df)} patents", f"{search_time_ms} ms")
            with m2:
                top_match_id = results_df["patent_id"].iloc[0] if not results_df.empty else "N/A"
                st.metric("Top Prior-Art Match", top_match_id)
            with m3:
                top_score = results_df["hybrid_score"].iloc[0] if not results_df.empty else 0.0
                st.metric("Highest Similarity", f"{top_score:.3f}")
            with m4:
                st.metric("Novelty Verdict", novelty["risk_label"].split()[0], f"{novelty['risk_score']}% collision risk")

            # Novelty Risk & Score Decomposition Visualizations
            st.markdown("---")
            col_gauge, col_radar = st.columns([1, 1])

            with col_gauge:
                st.subheader("🎯 Novelty Risk Gauge")
                # Plotly Gauge Indicator
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=novelty["risk_score"],
                    title={"text": f"Prior-Art Collision Risk: {novelty['risk_label']}", "font": {"size": 15}},
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": novelty["risk_color"]},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                            {"range": [40, 70], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"}
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": novelty["risk_score"]
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(f"💡 **Analysis:** {novelty['summary']}")

            with col_radar:
                st.subheader("📊 Retrieval Modality Breakdown")
                if not results_df.empty:
                    # Multi-bar comparison for top results
                    top_bar_df = results_df.head(min(5, len(results_df)))
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=top_bar_df["patent_id"],
                        y=top_bar_df["lexical_score"],
                        name="Lexical (BM25)",
                        marker_color="#3b82f6"
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=top_bar_df["patent_id"],
                        y=top_bar_df["semantic_score"],
                        name="Semantic (Dense)",
                        marker_color="#8b5cf6"
                    ))
                    if "rerank_score" in top_bar_df.columns and retrieval_mode == "neural_rerank":
                        fig_bar.add_trace(go.Bar(
                            x=top_bar_df["patent_id"],
                            y=top_bar_df["rerank_score"],
                            name="Cross-Encoder Re-rank",
                            marker_color="#10b981"
                        ))
                    fig_bar.add_trace(go.Bar(
                        x=top_bar_df["patent_id"],
                        y=top_bar_df["hybrid_score"],
                        name="Hybrid Final Score",
                        marker_color="#f59e0b"
                    ))
                    fig_bar.update_layout(
                        barmode="group",
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(range=[0, 1.05], title="Normalized Score"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Keyword Overlap & Concept Differential
            st.markdown("#### 🔬 Keyword Saliency & Concept Differentiation")
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.markdown("**Overlapping Prior-Art Concepts (Collision Points):**")
                if novelty.get("overlapping_keywords"):
                    badges = "".join([f"<span class='keyword-badge-pos'>⚠️ {kw}</span>" for kw in novelty["overlapping_keywords"]])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.info("No substantial technical term collisions detected.")
            with col_k2:
                st.markdown("**Distinctive Novel Elements (Potential Inventive Step):**")
                if novelty.get("distinctive_novel_keywords"):
                    badges = "".join([f"<span class='keyword-badge-novel'>✨ {kw}</span>" for kw in novelty["distinctive_novel_keywords"]])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.info("All query keywords are documented in the cited disclosure.")

            # Detailed Results Cards
            st.markdown("---")
            st.subheader(f"📑 Top {len(results_df)} Retrieved Prior-Art Patents")

            for rank, row in results_df.iterrows():
                with st.expander(f"#{rank+1} [{row['patent_id']}] {row['title']} — Hybrid Score: {row['hybrid_score']:.3f}", expanded=(rank == 0)):
                    col_info, col_metrics = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**CPC Category:** `{row.get('cpc_class', 'N/A')}` | **IPC Domain:** `{row.get('ipc_section', 'N/A')}`")
                        st.markdown(f"**Assignee:** `{row.get('assignee', 'Published Assignee')}` | **Filing Date:** `{row.get('filing_date', 'N/A')}`")
                        st.markdown(f"**Abstract:** {row['abstract']}")
                        if str(row.get("claims", "")).strip():
                            with st.expander("🔍 View Independent Claim 1"):
                                st.code(row["claims"], language="text")

                    with col_metrics:
                        st.markdown("##### Score Breakdown")
                        st.progress(float(row["hybrid_score"]), text=f"Hybrid Score: {row['hybrid_score']:.3f}")
                        st.progress(float(row["semantic_score"]), text=f"Semantic: {row['semantic_score']:.3f}")
                        st.progress(float(row["lexical_score"]), text=f"Lexical BM25: {row['lexical_score']:.3f}")
                        if "rerank_score" in row and retrieval_mode == "neural_rerank":
                            st.progress(float(row["rerank_score"]), text=f"Cross-Encoder: {row['rerank_score']:.3f}")

                        if str(row.get("source_url", "")).strip():
                            st.link_button("🌐 Open in Google Patents", row["source_url"], use_container_width=True)


# ==============================================================================
# TAB 2: 2D SEMANTIC PATENT MAP (LANDSCAPE VISUALIZER)
# ==============================================================================
with tabs[1]:
    st.subheader("🌐 2D Semantic Patent Landscape Projection")
    st.markdown(
        "Dimensionality reduction (**PCA**) projects high-dimensional dense embeddings into a 2D topographic "
        "technology space. Each patent is clustered by domain, with the active invention query marked."
    )

    plot_df = search_engine.patents.copy()
    if "pca_x" in plot_df.columns and "pca_y" in plot_df.columns:
        fig_map = px.scatter(
            plot_df,
            x="pca_x",
            y="pca_y",
            color="ipc_section",
            hover_data=["patent_id", "title", "cpc_class", "assignee"],
            title="Semantic Distribution of Indexed Patent Corpus",
            opacity=0.7,
            height=600,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        # If a search was run, plot query location
        if 'novelty' in locals() and novelty.get("query_coords"):
            qx, qy = novelty["query_coords"]
            fig_map.add_trace(go.Scatter(
                x=[qx],
                y=[qy],
                mode="markers+text",
                name="📍 Current Invention Query",
                marker=dict(size=18, color="#f59e0b", symbol="star", line=dict(color="white", width=2)),
                text=["YOUR INVENTION"],
                textposition="top center"
            ))

            # Highlight top retrieved match
            if 'results_df' in locals() and not results_df.empty:
                top_hit = results_df.iloc[0]
                fig_map.add_trace(go.Scatter(
                    x=[top_hit["pca_x"]],
                    y=[top_hit["pca_y"]],
                    mode="markers+text",
                    name="🎯 Top Prior-Art Hit",
                    marker=dict(size=14, color="#ef4444", symbol="diamond", line=dict(color="white", width=2)),
                    text=[f"TOP MATCH: {top_hit['patent_id']}"],
                    textposition="bottom center"
                ))

        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Principal Component 1"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Principal Component 2"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Insufficient records in current corpus to generate 2D projection.")


# ==============================================================================
# TAB 3: BENCHMARK & EVALUATION STUDIO
# ==============================================================================
with tabs[2]:
    st.subheader("📊 Information Retrieval (IR) Accuracy Benchmark Studio")
    st.markdown(
        "Evaluate retrieval precision, recall, and ranking effectiveness against standard ground-truth patent queries. "
        "Compares **BM25**, **SBERT**, **Hybrid Fusion**, **RRF**, and **Two-Stage Cross-Encoder**."
    )

    benchmark_path = "data/benchmark_queries.json"
    if os.path.exists(benchmark_path):
        if st.button("🚀 Run Live Benchmark & Alpha Parameter Sweep", type="primary"):
            with st.spinner("Executing benchmark across test queries..."):
                summary_df, sweep_df = evaluate_engine_benchmarks(search_engine, benchmark_path)
                st.session_state["benchmark_summary"] = summary_df
                st.session_state["benchmark_sweep"] = sweep_df

        if "benchmark_summary" in st.session_state:
            summary_df = st.session_state["benchmark_summary"]
            sweep_df = st.session_state["benchmark_sweep"]

            st.markdown("### 🏆 Comprehensive Retrieval Performance Metrics")
            st.dataframe(summary_df.style.highlight_max(axis=0, color="#1e3a8a"), use_container_width=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                # Precision@K Curve
                ks = [1, 5, 10]
                fig_pk = go.Figure()
                for method in summary_df.index:
                    y_vals = [summary_df.loc[method, f"P@{k}"] for k in ks]
                    fig_pk.add_trace(go.Scatter(x=ks, y=y_vals, mode="lines+markers", name=method))
                fig_pk.update_layout(
                    title="Precision @ K Comparison",
                    xaxis=dict(title="Cutoff (K)", tickvals=[1, 5, 10]),
                    yaxis=dict(title="Precision", range=[0, 1.05]),
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_pk, use_container_width=True)

            with col_g2:
                # Recall@K Curve
                fig_rk = go.Figure()
                for method in summary_df.index:
                    y_vals = [summary_df.loc[method, f"Recall@{k}"] if f"Recall@{k}" in summary_df.columns else 0.0 for k in [5, 10]]
                    fig_rk.add_trace(go.Scatter(x=[5, 10], y=y_vals, mode="lines+markers", name=method))
                fig_rk.update_layout(
                    title="Recall @ K Comparison",
                    xaxis=dict(title="Cutoff (K)", tickvals=[5, 10]),
                    yaxis=dict(title="Recall", range=[0, 1.05]),
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_rk, use_container_width=True)

            col_g3, col_g4 = st.columns(2)
            with col_g3:
                # MAP & MRR Comparison Bar
                fig_map = px.bar(
                    summary_df.reset_index(),
                    x="Method",
                    y=["MAP", "MRR"],
                    barmode="group",
                    title="Mean Average Precision (MAP) & MRR Ranking Quality",
                    height=360
                )
                fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_map, use_container_width=True)

            with col_g4:
                # Alpha Parameter Sweep Curve
                fig_sweep = px.line(
                    sweep_df,
                    x="Alpha",
                    y="MAP",
                    markers=True,
                    title="Fusion Weight (α) Sweep: Optimizing Lexical vs Semantic Balance",
                    height=360
                )
                fig_sweep.update_traces(line_color="#10b981", marker=dict(size=8))
                fig_sweep.update_layout(
                    xaxis_title="Alpha (Weight on BM25 Lexical)",
                    yaxis_title="Validation MAP",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_sweep, use_container_width=True)
    else:
        st.warning("Ground-truth benchmark query file not detected in `data/benchmark_queries.json`.")


# ==============================================================================
# TAB 4: MODEL FINE-TUNING & DOMAIN ADAPTATION STUDIO
# ==============================================================================
with tabs[3]:
    st.subheader("🚀 Model Training & Domain Adaptation Studio")
    st.markdown(
        "Fine-tune dense sentence transformer embeddings directly on patent claims, disclosures, and triplet pairs "
        "using **Multiple Negatives Ranking Loss (MNRL)** to adapt generic models to patent legalese."
    )

    col_tr1, col_tr2 = st.columns([1, 1])
    with col_tr1:
        st.markdown("#### Fine-Tuning Hyperparameters")
        train_epochs = st.slider("Training Epochs", 1, 5, 2)
        train_batch = st.select_slider("Batch Size", [2, 4, 8, 16], value=4)
        warmup_steps = st.slider("Warmup Steps", 1, 20, 5)

        pairs_path = "data/patent_training_pairs.json"
        if os.path.exists(pairs_path):
            with open(pairs_path) as f:
                pair_count = len(json.load(f))
            st.info(f"Loaded **{pair_count}** patent anchor-positive-negative triplets from `{pairs_path}`.")
        else:
            st.error("Training dataset `data/patent_training_pairs.json` missing.")

        if st.button("🔥 Start Patent Model Fine-Tuning", type="primary", use_container_width=True):
            with st.spinner("Fine-tuning SentenceTransformer with PyTorch on patent triplets..."):
                trainer = PatentModelTrainer(
                    base_model_name="sentence-transformers/all-MiniLM-L6-v2",
                    output_dir="models/patent_hybrid_finetuned"
                )
                train_result = trainer.train(
                    dataset_path=pairs_path,
                    epochs=train_epochs,
                    batch_size=train_batch,
                    warmup_steps=warmup_steps
                )
                st.session_state["train_result"] = train_result
                st.success(f"Fine-tuning completed in {train_result['elapsed_seconds']}s! Checkpoint saved to `{train_result['output_dir']}`.")
                st.rerun()

    with col_tr2:
        st.markdown("#### Checkpoint Status & Training History")
        if "train_result" in st.session_state:
            tr = st.session_state["train_result"]
            st.json(tr)
            # Plot loss convergence
            if tr.get("loss_curve"):
                loss_df = pd.DataFrame(tr["loss_curve"])
                fig_loss = px.line(loss_df, x="epoch", y="loss", markers=True, title="Training Loss Convergence")
                fig_loss.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_loss, use_container_width=True)
        elif has_finetuned:
            st.success("✅ **Fine-Tuned Patent Model Found:** `models/patent_hybrid_finetuned` is ready and selectable in the sidebar!")
        else:
            st.info("No local fine-tuned model checkpoint yet. Click 'Start Patent Model Fine-Tuning' to train.")


# ==============================================================================
# TAB 5: DATASET EXPLORER & INGESTION
# ==============================================================================
with tabs[4]:
    st.subheader("📁 Patent Corpus Management & Live Ingestion")

    # Ingestion Buttons & Uploaders
    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        st.markdown("#### Ingest Real USPTO Dataset from Hugging Face")
        st.markdown("Fetch 1,000 real USPTO patents across 8 CPC technology sectors.")
        if st.button("⚡ Download 1,000 Real USPTO Patents", use_container_width=True):
            with st.spinner("Downloading and processing real patent abstracts from Hugging Face..."):
                download_and_format(1000, "data/uspto_patents_large.csv")
                st.success("Successfully downloaded 1,000 real patents to `data/uspto_patents_large.csv`!")
                st.cache_resource.clear()
                st.rerun()

    with col_up2:
        st.markdown("#### Upload Custom Patent CSV / Portfolio")
        uploaded_file = st.file_uploader("Upload CSV containing (patent_id, title, abstract)", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file).fillna("")
            user_path = f"data/custom_{uploaded_file.name}"
            user_df.to_csv(user_path, index=False)
            st.success(f"Uploaded `{len(user_df)}` custom patents to `{user_path}`!")
            st.cache_resource.clear()

    st.markdown("---")
    st.subheader(f"📊 Active Corpus Explorer: `{corpus_choice}` ({len(search_engine.patents):,} patents)")

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        # Domain distribution donut chart
        if "ipc_section" in search_engine.patents.columns:
            domain_counts = search_engine.patents["ipc_section"].value_counts().reset_index()
            domain_counts.columns = ["Domain", "Count"]
            fig_donut = px.pie(domain_counts, names="Domain", values="Count", hole=0.5, title="Corpus Technological Sectors")
            fig_donut.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)

    with col_stat2:
        # Document word count histogram
        words_per_doc = search_engine.patents["text"].apply(lambda x: len(str(x).split()))
        fig_hist = px.histogram(words_per_doc, nbins=30, title="Document Word Count Distribution", labels={"value": "Words"})
        fig_hist.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Searchable Data Table
    st.markdown("#### Browse Corpus Records")
    search_term = st.text_input("Filter records by keyword:", placeholder="Filter by title, assignee, or CPC code...")
    display_df = search_engine.patents.copy()
    if search_term.strip():
        display_df = display_df[
            display_df["title"].str.contains(search_term, case=False, na=False) |
            display_df["abstract"].str.contains(search_term, case=False, na=False) |
            display_df["patent_id"].str.contains(search_term, case=False, na=False)
        ]

    cols_to_show = [c for c in ["patent_id", "title", "cpc_class", "ipc_section", "assignee", "filing_date"] if c in display_df.columns]
    st.dataframe(display_df[cols_to_show].head(100), use_container_width=True)

st.markdown("---")
st.caption(
    "Hybrid AI Patent Prior-Art Retrieval Platform | "
    "Combining Okapi BM25 Term Saliency, Sentence-Transformers Dense Cosine Geometry, and Deep Neural Cross-Attention."
)
