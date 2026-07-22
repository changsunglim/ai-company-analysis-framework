"""
Web UI for the AI Company Analysis Framework.

Run locally:
    streamlit run streamlit_app.py

Or deploy for free on Streamlit Community Cloud pointed at this repo/file.
Each visitor supplies their own OpenAI API key — nothing is stored server-side.
"""

import asyncio
import tempfile
from pathlib import Path

import streamlit as st
from openai import AuthenticationError, APIError, APIConnectionError

from src.pipeline import AnalysisPipeline

MODULES = [
    "financial_analysis",
    "news_sentiment",
    "competitive_position",
    "risk_assessment",
    "growth_outlook",
]
MODULE_LABELS = {
    "financial_analysis": "Financial Health",
    "news_sentiment": "News Sentiment",
    "competitive_position": "Competitive Position",
    "risk_assessment": "Risk Assessment",
    "growth_outlook": "Growth Outlook",
}

st.set_page_config(page_title="AI Company Analysis", page_icon="📊", layout="centered")

st.title("📊 AI Company Analysis")
st.caption(
    "Pulls financial data + news for a ticker and runs it through an LLM "
    "for a structured research report. Not financial advice."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Used only for this request, never stored or logged. "
        "Get one at platform.openai.com/api-keys.",
    )
    st.caption("A full report costs roughly $0.005–$0.02 with gpt-4o-mini.")
    selected_modules = st.multiselect(
        "Analysis modules",
        options=MODULES,
        default=MODULES,
        format_func=lambda m: MODULE_LABELS[m],
    )

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", placeholder="AAPL")
with col2:
    company = st.text_input("Company name (optional)", placeholder="Apple Inc")

run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first.")
    elif not ticker.strip():
        st.error("Enter a ticker.")
    elif not selected_modules:
        st.error("Pick at least one analysis module.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with st.spinner("Collecting data and running analysis (usually 30-90s)..."):
                    pipeline = AnalysisPipeline(api_key=api_key)
                    pipeline.reporter.output_dir = Path(tmp_dir)

                    report_path = asyncio.run(
                        pipeline.run(
                            company=company.strip() or ticker.strip(),
                            ticker=ticker.strip(),
                            modules=selected_modules,
                        )
                    )
                    report_text = Path(report_path).read_text(encoding="utf-8")

                st.success("Done.")
                st.download_button(
                    "Download report (.md)",
                    data=report_text,
                    file_name=Path(report_path).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.markdown(report_text)

            except AuthenticationError:
                st.error("OpenAI rejected that API key. Double-check it and try again.")
            except (APIConnectionError, APIError) as e:
                st.error(f"OpenAI API error: {e}")
            except RuntimeError as e:
                st.error(f"{e} Check the ticker is valid and try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
