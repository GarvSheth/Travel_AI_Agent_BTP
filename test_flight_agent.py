"""
Interactive test script for the flight agent.

Phase 1  — Duffel API search (always runs, no LLM needed)
Phase 2  — Full agent with preference clarification + LLM ranking
            (requires GEMINI_API_KEY / ANTHROPIC_API_KEY in .env)

Run:
    python test_flight_agent.py
"""
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()


# ── Phase 1: raw Duffel search ────────────────────────────────────────────────

def test_duffel_client():
    console.print(Panel("[bold cyan]Phase 1 — Duffel API raw search[/bold cyan]"))
    from services.flight_provider.duffel_client import DuffelClient, normalize_offer

    try:
        client = DuffelClient()
    except Exception as exc:
        console.print(f"[red]✗ DuffelClient init failed: {exc}[/red]")
        return False

    console.print("[green]✓ DuffelClient initialized[/green]")
    console.print("[dim]Searching LHR → JFK, 2026-12-15, 1 adult, economy …[/dim]")

    try:
        raw = client.search_flights(
            origin="LHR",
            destination="JFK",
            departure_date="2026-12-15",
            num_passengers=1,
            cabin_class="economy",
            limit=10,
        )
    except Exception as exc:
        console.print(f"[red]✗ Search failed: {exc}[/red]")
        return False

    console.print(f"[green]✓ Duffel returned {len(raw)} raw offers[/green]")
    if not raw:
        console.print("[yellow]No offers returned (route may have no results in test mode).[/yellow]")
        return True

    offers = [normalize_offer(o) for o in raw]

    table = Table(title="Normalized offers (top 5)", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Airline")
    table.add_column("Flight")
    table.add_column("Dep → Arr")
    table.add_column("Duration")
    table.add_column("Stops", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Cabin")

    for i, o in enumerate(offers[:5], 1):
        dep = (o["departure_time"] or "")[:16].replace("T", " ")
        arr = (o["arrival_time"] or "")[:16].replace("T", " ")
        table.add_row(
            str(i),
            o["airline"],
            o["flight_number"],
            f"{dep} → {arr}",
            o["duration"],
            str(o["stops"]),
            f"{o['price_usd']:.2f} {o['currency']}",
            o["cabin_class"],
        )

    console.print(table)
    return True


# ── Phase 2: full agent ───────────────────────────────────────────────────────

def test_full_agent():
    console.print(Panel("[bold cyan]Phase 2 — Full flight agent (clarify → search → rank)[/bold cyan]"))

    # Detect which LLM key is available
    llm_provider = None
    if os.getenv("GEMINI_API_KEY"):
        llm_provider = "gemini"
    elif os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_API_KEY").startswith("sk-ant-..."):
        llm_provider = "anthropic"
    elif os.getenv("OPENROUTER_API_KEY"):
        llm_provider = "openrouter"

    if not llm_provider:
        console.print(
            "[yellow]⚠  No LLM API key found in .env "
            "(GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY).\n"
            "   Add one to .env and re-run for the full ranking test.[/yellow]"
        )
        return

    console.print(f"[green]✓ Using LLM provider: {llm_provider}[/green]")

    api_key = (
        os.getenv("GEMINI_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY") or
        os.getenv("OPENROUTER_API_KEY")
    )
    from services.llm_client import initialize_client
    initialize_client(llm_provider, api_key)
    console.print("[green]✓ LLM client initialized[/green]")

    from agents.flight_agent import FlightSearchRequest, search_flights_for_trip, resume_flight_search

    request = FlightSearchRequest(
        origin="LHR",
        destination="JFK",
        departure_date="2026-12-15",
        num_passengers=1,
        preferences="looking for the cheapest option, flexible on timing",
    )

    console.print(f"[dim]Starting flight agent: {request.origin} → {request.destination}[/dim]")
    outcome = search_flights_for_trip(request)

    # Handle preference clarification loop
    # Use a preset answer for non-interactive demo mode
    DEMO_ANSWER = "economy class, direct flights only please, morning departure preferred"
    rounds = 0
    while outcome.needs_input and rounds < 3:
        rounds += 1
        console.print(f"\n[bold cyan]Agent:[/bold cyan] {outcome.question}")
        console.print(f"[bold yellow]You (demo):[/bold yellow] {DEMO_ANSWER}")
        outcome = resume_flight_search(outcome.thread_id, DEMO_ANSWER)

    if outcome.status == "error":
        console.print(f"[red]✗ Agent error: {outcome.error}[/red]")
        return

    if outcome.status == "complete" and outcome.result:
        result = outcome.result
        console.print(f"\n[green]✓ Search complete — {result.candidates_found} offers fetched[/green]")

        if result.warnings:
            for w in result.warnings:
                console.print(f"[yellow]⚠  {w}[/yellow]")

        if not result.flights:
            console.print("[yellow]No flights matched your preferences.[/yellow]")
            return

        table = Table(title=f"Top {len(result.flights)} recommended flights", show_lines=True)
        table.add_column("Rank", justify="right", width=4)
        table.add_column("Score", justify="right", width=5)
        table.add_column("Airline")
        table.add_column("Flight")
        table.add_column("Dep → Arr")
        table.add_column("Duration")
        table.add_column("Stops", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Why")

        for f in result.flights:
            dep = (f.departure_time or "")[:16].replace("T", " ")
            arr = (f.arrival_time or "")[:16].replace("T", " ")
            table.add_row(
                str(f.rank),
                str(f.match_score),
                f.airline,
                f.flight_number,
                f"{dep} → {arr}",
                f.duration,
                str(f.stops),
                f"{f.price_usd:.2f} {f.currency}",
                f.explanation[:60] + ("…" if len(f.explanation) > 60 else ""),
            )

        console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(
        Panel(
            "[bold]Flight Agent — Integration Test[/bold]\n"
            "Tests the Duffel API client and the full LangGraph flight workflow.",
            expand=False,
        )
    )

    ok = test_duffel_client()
    console.print()
    test_full_agent()

    console.print("\n[bold green]Done.[/bold green]")
