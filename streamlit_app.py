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

# Any OpenAI-compatible provider works. The curated cards cover the common ones;
# OpenRouter exposes a model field, and "Custom" takes a raw base URL + model so a
# visitor can point at anything (DeepSeek, Mistral, Together, a self-hosted endpoint…).
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_label": "Groq API key",
        "key_help": "Get a free key at console.groq.com/keys.",
        "key_placeholder": "gsk_...",
        "price": "Free",
        "tag": "Recommended",
        "recommended": True,
        "mono": "Gq",
        "chip": "#C2683B",
        "notes": [
            "No cost, ever",
            "Usually the fastest provider",
            "Low free-tier rate limit",
        ],
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "model": "llama-3.3-70b",
        "key_label": "Cerebras API key",
        "key_help": "Get a free key at cloud.cerebras.ai.",
        "key_placeholder": "csk-...",
        "price": "Free",
        "tag": "Fastest",
        "recommended": False,
        "mono": "Cb",
        "chip": "#2F8080",
        "notes": [
            "No cost on the free tier",
            "Extremely fast inference",
            "Daily token limits apply",
        ],
    },
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "key_label": "Gemini API key",
        "key_help": "Get a free key at aistudio.google.com/apikey.",
        "key_placeholder": "AIza...",
        "price": "Free",
        "tag": "Generous limits",
        "recommended": False,
        "mono": "Gm",
        "chip": "#4A72B0",
        "notes": [
            "No cost, ever",
            "Comparable quality to Groq",
            "Roomier free tier",
        ],
    },
    "OpenAI": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "key_label": "OpenAI API key",
        "key_help": "Get one at platform.openai.com/api-keys.",
        "key_placeholder": "sk-...",
        "price": "~$0.01",
        "tag": "Most reliable",
        "recommended": False,
        "mono": "AI",
        "chip": "#3F7D5A",
        "notes": [
            "Per report, on gpt-4o-mini",
            "No free-tier rate cap",
            "Requires a funded account",
        ],
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "model_editable": True,
        "key_label": "OpenRouter API key",
        "key_help": "Get a key at openrouter.ai/keys.",
        "key_placeholder": "sk-or-...",
        "price": "Free+",
        "tag": "100s of models",
        "recommended": False,
        "mono": "OR",
        "chip": "#7A6AA8",
        "notes": [
            "One key, hundreds of models",
            "Free and paid models",
            "Pick any model below",
        ],
    },
    "Custom": {
        "base_url": None,
        "model": None,
        "base_editable": True,
        "model_editable": True,
        "base_placeholder": "https://api.deepseek.com",
        "model_placeholder": "deepseek-chat",
        "key_label": "API key",
        "key_help": "Any OpenAI-compatible provider — DeepSeek, Mistral, Together, self-hosted, etc.",
        "key_placeholder": "your provider key",
        "price": "Any",
        "tag": "Bring your own",
        "recommended": False,
        "mono": "＋",
        "chip": "#8A8378",
        "notes": [
            "Any OpenAI-compatible endpoint",
            "Set the base URL + model",
            "DeepSeek, Mistral, Together…",
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
        position: relative;
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: 16px;
        padding: 1.1rem 1.2rem 1rem 1.2rem;
        margin-bottom: 0.55rem;
        min-height: 186px;
        overflow: hidden;
        will-change: transform;
        transition: transform 260ms cubic-bezier(0.34, 1.42, 0.64, 1),
                    box-shadow 260ms ease, border-color 200ms ease, background 220ms ease;
    }
    /* whole-column hover lifts the card — covers the button below it too */
    div[data-testid="stColumn"]:hover .provider-card,
    div[data-testid="column"]:hover .provider-card {
        transform: translateY(-6px);
        box-shadow: 0 20px 42px -24px rgba(60, 48, 18, 0.42);
        border-color: #D8CDB4;
    }
    .provider-card.selected {
        border-color: var(--accent);
        background: linear-gradient(180deg, var(--accent-soft), var(--surface) 62%);
        box-shadow: 0 0 0 1.5px var(--accent), 0 22px 46px -26px var(--accent-shadow);
        transform: translateY(-3px);
    }
    div[data-testid="stColumn"]:hover .provider-card.selected,
    div[data-testid="column"]:hover .provider-card.selected {
        transform: translateY(-8px);
    }
    /* accent hairline that sweeps across the top edge on select */
    .provider-card::after {
        content: "";
        position: absolute;
        left: 0; top: 0;
        height: 3px; width: 100%;
        background: var(--accent);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 320ms cubic-bezier(0.34, 1.42, 0.64, 1);
    }
    .provider-card.selected::after { transform: scaleX(1); }

    .pc-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.75rem; }
    .pc-mono {
        flex: 0 0 auto;
        width: 32px; height: 32px;
        border-radius: 10px;
        display: grid; place-items: center;
        font-size: 0.72rem; font-weight: 600; color: #fff;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18),
                    0 3px 8px -3px rgba(0, 0, 0, 0.35);
    }
    .pc-name { font-weight: 600; font-size: 1.02rem; color: var(--ink); line-height: 1.15; }
    .pc-tag {
        font-size: 0.62rem; font-weight: 600; letter-spacing: 0.09em;
        text-transform: uppercase; color: var(--accent); margin-top: 0.15rem;
    }
    .provider-card .price {
        font-variant-numeric: tabular-nums;
        font-size: 1.32rem; font-weight: 600; color: var(--ink);
        margin-bottom: 0.55rem;
    }
    .provider-card .price span {
        font-size: 0.72rem; font-weight: 400; color: var(--muted); margin-left: 0.28rem;
    }
    .provider-card ul {
        margin: 0; padding-left: 1rem;
        color: var(--muted); font-size: 0.8rem; line-height: 1.6;
    }

    @media (prefers-reduced-motion: reduce) {
        .provider-card, .provider-card::after { transition: none; }
        div[data-testid="stColumn"]:hover .provider-card,
        div[data-testid="column"]:hover .provider-card,
        .provider-card.selected { transform: none; }
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
        name for name, p in PROVIDERS.items() if p.get("recommended")
    )

provider_items = list(PROVIDERS.items())
for row_start in range(0, len(provider_items), 3):
    cols = st.columns(3)
    for col, (name, p) in zip(cols, provider_items[row_start:row_start + 3]):
        with col:
            is_selected = st.session_state.provider_name == name
            notes_html = "".join(f"<li>{n}</li>" for n in p["notes"])
            tag_html = f'<div class="pc-tag">{p["tag"]}</div>' if p.get("tag") else ""
            card_class = "provider-card selected" if is_selected else "provider-card"
            st.markdown(
                f'<div class="{card_class}">'
                f'<div class="pc-head">'
                f'<div class="pc-mono" style="background:{p["chip"]}">{p["mono"]}</div>'
                f'<div><div class="pc-name">{name}</div>{tag_html}</div>'
                f'</div>'
                f'<div class="price">{p["price"]}<span>/ report</span></div>'
                f'<ul>{notes_html}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                "✓ Selected" if is_selected else "Select",
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

# Providers that let the visitor point at a specific endpoint / model.
eff_base_url = provider["base_url"]
eff_model = provider["model"]
if provider.get("base_editable"):
    ec1, ec2 = st.columns(2)
    with ec1:
        eff_base_url = st.text_input(
            "API base URL", placeholder=provider.get("base_placeholder", ""),
        ).strip() or None
    with ec2:
        eff_model = st.text_input(
            "Model", placeholder=provider.get("model_placeholder", ""),
        ).strip() or None
elif provider.get("model_editable"):
    eff_model = st.text_input(
        "Model", value=provider["model"], help="Any model ID this provider serves.",
    ).strip() or provider["model"]

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
    elif provider.get("base_editable") and not eff_base_url:
        st.error("Enter the API base URL for your custom provider.")
    elif provider.get("model_editable") and not eff_model:
        st.error("Enter the model ID for your provider.")
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
                        base_url=eff_base_url,
                        model=eff_model,
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
