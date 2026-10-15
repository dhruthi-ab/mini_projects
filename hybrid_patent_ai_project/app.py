import streamlit as st
from search_engine import HybridPatentSearch

st.set_page_config(page_title="Hybrid Patent AI", page_icon="🔎", layout="wide")

st.title("🔎 Hybrid AI-Based Patent Prior-Art Retrieval")
st.write(
    "Search the included demonstration dataset using lexical and semantic similarity. "
    "The included records are synthetic examples, not real patents."
)

@st.cache_resource
def load_search_engine():
    return HybridPatentSearch("data/patents_demo.csv")

with st.spinner("Loading the search engine (the first run may download the embedding model)..."):
    try:
        search_engine = load_search_engine()
    except Exception as exc:
        st.error("Could not load the search engine. Check your internet connection and installed dependencies.")
        st.exception(exc)
        st.stop()

st.subheader("Describe your invention")
title = st.text_input("Invention title", placeholder="Example: Smart hydration bottle")
description = st.text_area(
    "Invention description",
    placeholder="Describe how it works, its technical features, and its purpose...",
    height=150,
)
lexical_weight = st.slider("Lexical weight", 0.0, 1.0, 0.5, 0.1)
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search prior art", type="primary"):
    if not title.strip() or not description.strip():
        st.warning("Please enter both a title and a description.")
    else:
        query = f"{title.strip()} {description.strip()}"
        results = search_engine.search(query, top_k=top_k, lexical_weight=lexical_weight)
        st.subheader("Potentially relevant documents")
        if results.empty:
            st.info("No results were returned.")
        else:
            for _, row in results.iterrows():
                with st.expander(f"{row['patent_id']} — {row['title']}"):
                    st.write("**Abstract**")
                    st.write(row["abstract"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Lexical score", f"{row['lexical_score']:.3f}")
                    c2.metric("Semantic score", f"{row['semantic_score']:.3f}")
                    c3.metric("Hybrid score", f"{row['hybrid_score']:.3f}")
                    if str(row.get("source_url", "")).strip():
                        st.link_button("View source document", row["source_url"])

st.divider()
st.caption(
    "Important: the bundled CSV contains synthetic demonstration records. "
    "Similarity scores are ranking signals, not probabilities or legal conclusions. "
    "Use a real patent-retrieval benchmark and relevance judgments for research evaluation."
)
