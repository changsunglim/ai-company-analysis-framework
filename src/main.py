"""
CLI entry point.

Usage:
    python -m src.main AAPL
    python -m src.main TSLA --company "Tesla Inc"
    python -m src.main 005930.KS --company "Samsung"
    python -m src.main AAPL --modules financial_analysis news_sentiment
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
        description="AI Company Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                             Analyze Apple
  %(prog)s MSFT --company "Microsoft"       Custom company name
  %(prog)s 005930.KS --company "Samsung"    Korean stocks
  %(prog)s AAPL --modules financial_analysis news_sentiment
        """,
    )

    parser.add_argument("ticker", help="Stock ticker (e.g. AAPL, 005930.KS)")
    parser.add_argument("--company", default=None, help="Company name for news search")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Config file path"
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
        help="Specific modules to run",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")

    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()
    company = args.company or args.ticker

    console.print(
        Panel.fit(
            f"[bold blue]AI Company Analysis Framework[/bold blue]\n"
            f"[dim]Target: [bold]{company}[/bold] ({args.ticker})[/dim]",
            border_style="blue",
        )
    )

    try:
        pipeline = AnalysisPipeline(config_path=args.config)

        if args.output_dir:
            pipeline.reporter.output_dir = Path(args.output_dir)
            pipeline.reporter.output_dir.mkdir(parents=True, exist_ok=True)

        report_path = await pipeline.run(
            company=company, ticker=args.ticker, modules=args.modules,
        )

        console.print(
            Panel.fit(
                f"[bold green]Done![/bold green]\n\n"
                f"Report: [underline]{report_path}[/underline]",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
