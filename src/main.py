"""
CLI entry point for the AI Company Analysis Framework.

Usage:
    python -m src.main AAPL                          # Analyze Apple
    python -m src.main TSLA --company "Tesla Inc"    # Custom company name
    python -m src.main 005930.KS --company "Samsung" # Korean stocks
    python -m src.main AAPL --modules financial news # Specific modules
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.pipeline import AnalysisPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Company Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                             Analyze Apple Inc.
  %(prog)s MSFT --company "Microsoft"       Use custom company name
  %(prog)s 005930.KS --company "Samsung"    Korean stock market
  %(prog)s AAPL --modules financial news    Run specific modules only
  %(prog)s GOOGL --config custom.yaml       Use custom config
        """,
    )

    parser.add_argument(
        "ticker",
        help="Stock ticker symbol (e.g., AAPL, TSLA, 005930.KS)",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Company name for news search (default: uses ticker)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=[
            "financial_analysis",
            "news_sentiment",
            "competitive_position",
            "risk_assessment",
            "growth_outlook",
        ],
        default=None,
        help="Specific analysis modules to run",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for reports",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    args = parse_args()
    company = args.company or args.ticker

    # Display banner
    console.print(
        Panel.fit(
            f"[bold blue]AI Company Analysis Framework[/bold blue]\n"
            f"[dim]Analyzing: [bold]{company}[/bold] ({args.ticker})[/dim]",
            border_style="blue",
        )
    )

    try:
        # Initialize and run pipeline
        pipeline = AnalysisPipeline(config_path=args.config)

        if args.output_dir:
            pipeline.reporter.output_dir = Path(args.output_dir)
            pipeline.reporter.output_dir.mkdir(parents=True, exist_ok=True)

        report_path = await pipeline.run(
            company=company,
            ticker=args.ticker,
            modules=args.modules,
        )

        console.print(
            Panel.fit(
                f"[bold green]Analysis Complete![/bold green]\n\n"
                f"Report saved to: [underline]{report_path}[/underline]",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
