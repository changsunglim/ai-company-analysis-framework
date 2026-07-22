"""
Web UI for the AI Company Analysis Framework.

Run locally:
    streamlit run streamlit_app.py

Or deploy for free on Streamlit Community Cloud pointed at this repo/file.
Each visitor supplies their own API key — nothing is stored server-side.
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

# Matches the presets documented in .env.example
PROVIDERS = {
    "OpenAI (paid, ~$0.005-0.02/run)": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "key_label": "OpenAI API key",
        "key_help": "Get one at platform.openai.com/api-keys.",
        "key_placeholder": "sk-...",
    },
    "Groq free tier (free, fast)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_label": "Groq API key",
        "key_help": "Get a free key at console.groq.com/keys.",
        "key_placeholder": "gsk_...",
    },
    "Gemini free tier (free)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "key_label": "Gemini API key",
        "key_help": "Get a free key at aistudio.google.com/apikey.",
        "key_placeholder": "AIza...",
    },
}

st.set_page_config(page_title="AI Company Analysis", page_icon="📊", layout="centered")

st.title("📊 AI Company Analysis")
st.caption(
    "Pulls financial data + news for a ticker and runs it through an LLM "
    "for a structured research report. Not financial advice."
)

with st.sidebar:
    st.header("Settings")
    provider_name = st.selectbox("Provider", options=list(PROVIDERS.keys()))
    provider = PROVIDERS[provider_name]

    api_key = st.text_input(
        provider["key_label"],
        type="password",
        help=f"Used only for this request, never stored or logged. {provider['key_help']}",
        placeholder=provider["key_placeholder"],
    )
    if provider["base_url"] is None:
        st.caption("A full report costs roughly $0.005–$0.02 with gpt-4o-mini.")
    else:
        st.caption("Free tier — report costs $0. May be slower or rate-limited.")

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
        st.error(f"Enter your {provider['key_label']} in the sidebar first.")
    elif not ticker.strip():
        st.error("Enter a ticker.")
    elif not selected_modules:
        st.error("Pick at least one analysis module.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with st.spinner("Collecting data and running analysis (usually 30-90s)..."):
                    pipeline = AnalysisPipeline(
                        api_key=api_key,
                        base_url=provider["base_url"],
                        model=provider["model"],
                    )
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
                st.error(f"{provider_name.split(' (')[0]} rejected that API key. Double-check it and try again.")
            except (APIConnectionError, APIError) as e:
                st.error(f"API error: {e}")
            except RuntimeError as e:
                st.error(f"{e} Check the ticker is valid and try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
