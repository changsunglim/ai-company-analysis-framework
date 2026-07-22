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

# Matches the presets documented in .env.example. Groq listed first (fastest free option).
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_label": "Groq API key",
        "key_help": "Get a free key at console.groq.com/keys.",
        "key_placeholder": "gsk_...",
        "price": "$0",
        "recommended": True,
        "notes": [
            "No cost, ever",
            "Usually the fastest provider",
            "Low free-tier rate limit — may pause between modules",
        ],
    },
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "key_label": "Gemini API key",
        "key_help": "Get a free key at aistudio.google.com/apikey.",
        "key_placeholder": "AIza...",
        "price": "$0",
        "recommended": False,
        "notes": [
            "No cost, ever",
            "Comparable quality to Groq",
            "Worth trying if Groq is rate-limited",
        ],
    },
    "OpenAI": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "key_label": "OpenAI API key",
        "key_help": "Get one at platform.openai.com/api-keys.",
        "key_placeholder": "sk-...",
        "price": "~$0.005–0.02",
        "recommended": False,
        "notes": [
            "Per report, on gpt-4o-mini",
            "No free-tier rate limit",
            "Requires a funded OpenAI account",
        ],
    },
}

st.set_page_config(page_title="AI Company Analysis", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Outfit:wght@400;500;600&display=swap');

    :root {
        --accent: #A9790A;
        --accent-soft: rgba(169, 121, 10, 0.10);
        --accent-shadow: rgba(169, 121, 10, 0.18);
        --ink: #211F1C;
        --muted: #716C63;
        --surface: #FFFFFF;
        --surface-border: #E7E2D6;
    }

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 { font-family: 'Fraunces', serif; letter-spacing: -0.01em; }

    /* ---- hero ---- */
    .hero-eyebrow {
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.6rem;
        line-height: 1.08;
        letter-spacing: -0.02em;
        margin: 0 0 0.8rem 0;
        text-wrap: balance;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 62ch;
        margin-bottom: 0;
    }
    .section-label {
        font-size: 0.76rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 2.4rem 0 0.9rem 0;
    }

    /* ---- provider cards ---- */
    .provider-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: 14px;
        padding: 1.25rem 1.3rem 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
        transition: border-color 180ms ease, box-shadow 180ms ease;
    }
    .provider-card.selected {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent), 0 8px 24px -14px var(--accent-shadow);
    }
    .provider-card .badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-soft);
        border-radius: 4px;
        padding: 0.18rem 0.5rem;
        margin-bottom: 0.55rem;
    }
    .provider-card .name {
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }
    .provider-card .price {
        font-variant-numeric: tabular-nums;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.7rem;
    }
    .provider-card .price span {
        font-size: 0.78rem;
        font-weight: 400;
        color: var(--muted);
        margin-left: 0.3rem;
    }
    .provider-card ul {
        margin: 0;
        padding-left: 1.05rem;
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.65;
    }

    /* ---- disclaimer ---- */
    .disclaimer {
        border-left: 2px solid var(--accent);
        background: var(--surface);
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.15rem;
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.65;
        margin: 1.6rem 0 2.2rem 0;
    }
    .disclaimer b { color: var(--ink); font-weight: 600; }

    /* ---- footer ---- */
    .app-footer {
        margin-top: 3rem;
        padding-top: 1.4rem;
        border-top: 1px solid var(--surface-border);
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.6;
    }
    .app-credit {
        margin-top: 0.9rem;
        color: var(--muted);
        font-size: 0.78rem;
    }
    .app-credit a {
        color: var(--ink);
        font-weight: 500;
        text-decoration: none;
        border-bottom: 1px solid var(--surface-border);
    }
    .app-credit a:hover { border-bottom-color: var(--accent); color: var(--accent); }

    /* ---- widget polish ---- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: scale(0.98);
    }
    div[data-testid="stTextInput"] input {
        border-radius: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-eyebrow">Company Research</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AI-powered company analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Pulls financial data and recent news for a ticker, then runs it through '
    "an LLM for a structured research report — financial health, sentiment, competitive position, "
    "risk, and growth outlook.</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="disclaimer">'
    "<b>Not financial advice.</b> This tool generates analysis using AI language models and public "
    "data sources, for informational purposes only. It does not constitute financial, investment, "
    "legal, or tax advice, and we accept no responsibility or liability for any financial decisions "
    "or outcomes based on its output. Verify data independently and consult a licensed financial "
    "advisor before making investment decisions."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Provider</div>', unsafe_allow_html=True)

if "provider_name" not in st.session_state:
    st.session_state.provider_name = next(
        name for name, p in PROVIDERS.items() if p["recommended"]
    )

cols = st.columns(3)
for col, (name, p) in zip(cols, PROVIDERS.items()):
    with col:
        is_selected = st.session_state.provider_name == name
        badge_html = '<div class="badge">Recommended</div>' if p["recommended"] else ""
        notes_html = "".join(f"<li>{n}</li>" for n in p["notes"])
        card_class = "provider-card selected" if is_selected else "provider-card"
        st.markdown(
            f'<div class="{card_class}">{badge_html}'
            f'<div class="name">{name}</div>'
            f'<div class="price">{p["price"]}<span>/ report</span></div>'
            f'<ul>{notes_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Selected" if is_selected else f"Choose {name}",
            key=f"select_{name}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
            disabled=is_selected,
        ):
            st.session_state.provider_name = name
            st.rerun()

provider_name = st.session_state.provider_name
provider = PROVIDERS[provider_name]

st.markdown('<div class="section-label">Credentials</div>', unsafe_allow_html=True)
api_key = st.text_input(
    provider["key_label"],
    type="password",
    help=f"Used only for this request, never stored or logged. {provider['key_help']}",
    placeholder=provider["key_placeholder"],
)

with st.expander("Analysis modules", expanded=False):
    selected_modules = st.multiselect(
        "Included in the report",
        options=MODULES,
        default=MODULES,
        format_func=lambda m: MODULE_LABELS[m],
        label_visibility="collapsed",
    )

st.markdown('<div class="section-label">Company</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", placeholder="AAPL")
with col2:
    company = st.text_input("Company name (optional)", placeholder="Apple Inc")

run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

if run_clicked:
    if not api_key:
        st.error(f"Enter your {provider['key_label']} first.")
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
                st.error(f"{provider_name} rejected that API key. Double-check it and try again.")
            except (APIConnectionError, APIError) as e:
                st.error(f"API error: {e}")
            except RuntimeError as e:
                st.error(f"{e} Check the ticker is valid and try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.markdown(
    '<div class="app-footer">AI Company Analysis Framework — analysis is generated by '
    "third-party language models and may be inaccurate or incomplete. Not financial advice. "
    "No accountability is held for financial decisions or outcomes made using this tool."
    '<div class="app-credit">Built by '
    '<a href="https://github.com/changsunglim" target="_blank">Isaac Lim</a>'
    "</div></div>",
    unsafe_allow_html=True,
)
