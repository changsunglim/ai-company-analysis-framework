"""
Prompt templates for each analysis module.
Spent a while tuning these to get good structured output.
"""

SYSTEM_PROMPT = """You are a senior equity research analyst with expertise in
financial analysis, competitive strategy, and market trends. You produce
clear, data-driven analysis backed by specific numbers and evidence.

Guidelines:
- Always cite specific data points from the provided context
- Distinguish between facts (from data) and your analytical interpretation
- Flag any data gaps or areas where more research is needed
- Use professional but accessible language
- Structure your analysis with clear headers and bullet points"""


ANALYSIS_PROMPTS = {
    "financial_analysis": {
        "name": "Financial Health Analysis",
        "prompt": """Analyze the following financial data for {company}.

{context}

Provide a comprehensive financial health assessment covering:

1. **Profitability**: Revenue trends, margins, and earnings quality
2. **Balance Sheet Strength**: Debt levels, liquidity, and asset quality
3. **Valuation**: Current valuation vs. historical and sector averages
4. **Cash Flow**: Free cash flow generation and capital allocation

For each area, assign a rating: STRONG / ADEQUATE / WEAK / INSUFFICIENT DATA

End with a 2-3 sentence overall financial health summary.""",
    },
    "news_sentiment": {
        "name": "News Sentiment Analysis",
        "prompt": """Analyze the following recent news articles about {company}.

{context}

Provide:

1. **Overall Sentiment**: POSITIVE / MIXED / NEGATIVE with confidence (HIGH/MEDIUM/LOW)
2. **Key Themes**: Top 3-5 recurring themes across articles
3. **Catalysts**: Any upcoming events or catalysts mentioned
4. **Risks Identified**: Any risks or concerns flagged in coverage
5. **Media Narrative**: What story is the media telling about this company?

Support each point with specific article references.""",
    },
    "competitive_position": {
        "name": "Competitive Position Analysis",
        "prompt": """Analyze the competitive landscape for {company} based on the following data.

{context}

Provide:

1. **Market Position**: Where does {company} stand vs. key competitors?
2. **Competitive Advantages**: What moats or advantages does {company} have?
3. **Competitive Threats**: Key threats from competitors or new entrants
4. **Key Differentiators**: What makes {company} unique in its market?
5. **Competitive Trend**: Is {company}'s competitive position STRENGTHENING / STABLE / WEAKENING?

Use specific metrics to support your comparisons.""",
    },
    "risk_assessment": {
        "name": "Risk Assessment",
        "prompt": """Based on all available data for {company}, provide a comprehensive risk assessment.

{context}

Identify and analyze risks across these categories:

1. **Financial Risks**: Balance sheet, cash flow, or profitability risks
2. **Market Risks**: Sector headwinds, valuation risks, macro factors
3. **Operational Risks**: Execution, management, or business model risks
4. **Regulatory Risks**: Legal, compliance, or policy change risks
5. **Competitive Risks**: Market share, disruption, or pricing risks

For each risk:
- Severity: HIGH / MEDIUM / LOW
- Probability: HIGH / MEDIUM / LOW
- Timeframe: NEAR-TERM (<1yr) / MEDIUM-TERM (1-3yr) / LONG-TERM (3yr+)

End with the top 3 risks an investor should monitor most closely.""",
    },
    "growth_outlook": {
        "name": "Growth Outlook",
        "prompt": """Based on all available data for {company}, assess the growth outlook.

{context}

Analyze:

1. **Revenue Growth Drivers**: What will drive future revenue?
2. **Market Opportunity**: Total addressable market and penetration
3. **Growth Investments**: R&D, capex, and strategic initiatives
4. **Growth Sustainability**: Are current growth rates sustainable?
5. **Expansion Vectors**: New markets, products, or geographies

Provide:
- 1-Year Outlook: ACCELERATING / STABLE / DECELERATING
- 3-Year Outlook: BULLISH / NEUTRAL / BEARISH

End with a confidence level (HIGH/MEDIUM/LOW) for your assessment.""",
    },
    "executive_summary": {
        "name": "Executive Summary",
        "prompt": """You are writing the executive summary for a comprehensive
company analysis report on {company}.

Here are the individual analysis sections:

{context}

Write a cohesive executive summary (300-500 words) that:

1. Opens with a one-sentence thesis on {company}
2. Highlights the 3 most important findings
3. Addresses the biggest risk and biggest opportunity
4. Provides an overall assessment: BULLISH / NEUTRAL / BEARISH
5. Ends with 2-3 key metrics to watch going forward

This should work as a standalone overview for a busy reader.""",
    },
}


class PromptManager:
    """Manages prompt templates."""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.templates = ANALYSIS_PROMPTS

    def get_analysis_prompt(
        self, module: str, company: str, context: str,
    ) -> dict[str, str]:
        """Build prompt for an analysis module."""
        if module not in self.templates:
            raise ValueError(
                f"Unknown module: {module}. "
                f"Available: {list(self.templates.keys())}"
            )

        template = self.templates[module]
        user_prompt = template["prompt"].format(
            company=company, context=context
        )

        return {
            "system": self.system_prompt,
            "user": user_prompt,
            "module_name": template["name"],
        }

    def get_available_modules(self) -> list[str]:
        return list(self.templates.keys())

    def estimate_prompt_tokens(self, module: str, context_tokens: int) -> int:
        """Rough token estimate for cost prediction."""
        # template ~200 tokens, system ~150 tokens
        return 200 + 150 + context_tokens
