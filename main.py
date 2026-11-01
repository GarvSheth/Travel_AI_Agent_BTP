"""Terminal version of the Travel AI Agent (same flow as streamlit_app.py).

    trip chat (input pipeline) -> hotel workflow (location / star questions)
    -> top hotels -> live room prices -> prebook -> guest details -> book
    -> booking confirmation (GET /bookings/{id}) + PDF saved to disk

Run:
    python main.py                 # PDF saved in the current folder
    python main.py --pdf-dir out   # PDF saved in ./out
"""
from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import APP_NAME, APP_VERSION
from services.credential_manager import CredentialManager
from services.llm_client import initialize_client

console = Console()


class Quit(Exception):
    """User chose to leave (q / Ctrl+C / Ctrl+D)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="travel-agent",
        description=f"{APP_NAME} v{APP_VERSION}: plan and book a hotel from the terminal",
    )
    parser.add_argument("--pdf-dir", default=".", help="folder for the booking confirmation PDF")
    return parser.parse_args(argv)


def bootstrap_credentials() -> str:
    """Return the Gemini key; stop with a clear message if any key is missing."""
    load_dotenv()
    cm = CredentialManager(provider="gemini")
    missing = cm.load_from_environment()
    if not os.getenv("NUITEE_API_KEY", "").strip():
        missing.append("NUITEE_API_KEY")
    if missing:
        names = "\n".join(f"  - {name}" for name in missing)
        raise SystemExit(f"Missing credentials. Set these in .env:\n{names}")
    return cm.get_credential("GEMINI_API_KEY")


# ── Small input helpers ───────────────────────────────────────────────────────

def ask(prompt: str) -> str:
    try:
        return console.input(f"[bold yellow]{prompt}[/bold yellow]").strip()
    except (KeyboardInterrupt, EOFError):
        raise Quit from None


def agent_says(text: str) -> None:
    console.print(f"\n[bold cyan]Agent:[/bold cyan] {text}")


def choose(prompt: str, count: int, extra: tuple[str, ...] = ("q",)) -> int | str:
    """A number 1..count, or one of the ``extra`` letters (q always quits)."""
    while True:
        answer = ask(prompt).lower()
        if answer == "q":
            raise Quit
        if answer in extra:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer)
        console.print(f"[red]Enter a number from 1 to {count}"
                      + "".join(f", or '{e}'" for e in extra) + ".[/red]")


def money(amount, currency) -> str:
    return "—" if amount is None else f"{amount:,.0f} {currency or ''}".strip()


# ── Steps ─────────────────────────────────────────────────────────────────────

def collect_trip():
    """Input pipeline: chat until every required trip detail is known."""
    from workflows.input_pipeline import build_input_graph, start_conversation, submit_answer

    graph, thread = build_input_graph(), f"input-{uuid.uuid4().hex}"
    console.print("[bold]Tell me about your trip[/bold] "
                  "(e.g. Plan a trip to Jaipur for 3 days from Delhi)\n")
    prompt = ""
    while not prompt:
        prompt = ask("You: ")
    turn = start_conversation(graph, thread, prompt)
    while not turn.complete:
        agent_says(turn.question)
        turn = submit_answer(graph, thread, ask("You: "))
    return turn.payload


def find_hotels(payload):
    """Hotel workflow: answer its questions until recommendations are ready."""
    from agents.hotel_agent import hotel_request_from_trip, resume_hotel_search, search_hotels

    request = hotel_request_from_trip(payload)
    with console.status(f"Hotel agent working on {request.city}…"):
        outcome = search_hotels(request)
    while outcome.needs_input:
        agent_says(outcome.question)
        answer = ask("You: ")
        with console.status("Searching and ranking hotels…"):
            outcome = resume_hotel_search(outcome.thread_id, answer)
    if outcome.status == "error":
        raise SystemExit(f"Sorry, I couldn't find hotels: {outcome.error}")
    return outcome.result


def show_hotels(result) -> None:
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    near = f" near {result.location.label.split(',')[0]}" if result.location else ""
    table = Table(title=f"Top {len(result.hotels)} of {result.candidates_found} hotels{near}")
    for col in ("#", "Hotel", "Stars", "Guest rating", "Distance", "Match", "Why it matches"):
        table.add_column(col, overflow="fold")
    for h in result.hotels:
        table.add_row(
            str(h.rank), h.name,
            f"{h.stars:g}★" if h.stars else "—",
            f"{h.guest_rating:g}/10" if h.guest_rating is not None else "—",
            f"{h.distance_km:.1f} km" if h.distance_km is not None else "—",
            f"{h.match_score}/100", h.explanation or "—",
        )
    console.print(table)


def show_offers(hotel, rates) -> None:
    stay = rates.stay
    table = Table(title=f"Prices at {hotel.name} · {stay.check_in} → {stay.check_out} · "
                        f"{stay.nights} night(s) · {stay.adults} adult(s) · {stay.rooms} room(s)")
    for col in ("#", "Room", "Board", "Refundable", "Total", "≈ per night"):
        table.add_column(col, overflow="fold")
    for i, o in enumerate(rates.offers, start=1):
        table.add_row(str(i), o.room_name, o.board or "—",
                      {True: "✓", False: "✗"}.get(o.refundable, "?"),
                      money(o.total, o.currency),
                      money(o.total / stay.nights, o.currency) if stay.nights else "—")
    console.print(table)
    if rates.sandbox:
        console.print("[yellow]Sandbox prices: test data, not bookable for real.[/yellow]")


def pick_and_prebook(result):
    """Hotel -> live prices -> offer -> prebook. Returns a PrebookResult."""
    from agents.hotel_agent import HotelAgent
    from services.hotel_provider.nuitee_client import NuiteeAPIError

    if result.stay is None:
        raise SystemExit("Trip dates are missing, so prices can't be checked.")
    agent = HotelAgent()
    while True:
        show_hotels(result)
        n = choose("Pick a hotel for live prices (number, or q to quit): ", len(result.hotels))
        hotel = result.hotels[n - 1]
        try:
            with console.status(f"Checking live prices for {hotel.name}…"):
                rates = agent.get_room_offers(hotel.hotel_id, result.stay)
        except (NuiteeAPIError, ValueError) as exc:
            console.print(f"[red]Couldn't get prices for {hotel.name}: {exc}[/red]")
            continue
        if not rates.offers:
            console.print("[yellow]No rooms available for these dates.[/yellow]")
            continue
        show_offers(hotel, rates)
        pick = choose("Prebook which room? (number, b = back to hotels, q = quit): ",
                      len(rates.offers), extra=("b",))
        if pick == "b":
            continue
        offer = rates.offers[pick - 1]
        try:
            with console.status(f"Holding {offer.room_name}…"):
                return agent.prebook(offer.offer_id, hotel.name)
        except (NuiteeAPIError, ValueError) as exc:
            console.print(f"[red]Prebook failed: {exc} Pick the hotel again for fresh "
                          "prices.[/red]")


def collect_guest():
    from agents.hotel_agent import GuestDetails

    console.print("\n[bold]Guest details[/bold] (used for every room)")
    values = {}
    for key, label in (("first", "First name: "), ("last", "Last name: "),
                       ("email", "Email: "), ("phone", "Phone: ")):
        while True:
            value = ask(label)
            if value and (key != "email" or "@" in value):
                values[key] = value
                break
            console.print("[red]Please enter a valid value.[/red]")
    return GuestDetails(values["first"], values["last"], values["email"], values["phone"])


def book(prebook):
    """Show the held room, take guest details, confirm, book. Returns BookingResult."""
    from agents.hotel_agent import HotelAgent
    from services.hotel_provider.nuitee_client import NuiteeAPIError

    console.print(Panel(
        f"[bold]{prebook.room_name}[/bold] at {prebook.hotel_name}\n"
        f"{prebook.check_in} → {prebook.check_out} · {prebook.board or 'board unknown'} · "
        f"{prebook.rooms} room(s)\nTotal: [green]{money(prebook.price, prebook.currency)}[/green]"
        f" · refundable: {({True: 'yes', False: 'no'}).get(prebook.refundable, 'unknown')}",
        title="Room held", border_style="cyan"))
    if prebook.price_changed:
        console.print("[yellow]The price changed since the search; the total above is new.[/yellow]")
    if prebook.terms_changed:
        console.print("[yellow]The cancellation policy or meal plan changed.[/yellow]")

    guest = collect_guest()
    if prebook.sandbox:
        console.print("[yellow]Sandbox key: the booking is simulated, nothing is charged.[/yellow]")
        confirm = ask("Book now? (y/n): ").lower() == "y"
    else:
        console.print("[bold red]This makes a real booking and charges the card stored on "
                      "your Nuitee account.[/bold red]")
        confirm = ask("Type 'yes' to book: ").lower() == "yes"
    if not confirm:
        raise Quit

    try:
        with console.status("Booking…"):
            return HotelAgent().book(prebook, guest)
    except (NuiteeAPIError, ValueError) as exc:
        raise SystemExit(f"Sorry, the booking didn't go through: {exc}") from None


def confirm_and_save(booking, pdf_dir: Path) -> None:
    """Retrieve the booking record, print the confirmation and save the PDF."""
    from agents.hotel_agent import get_booking_details
    from agents.hotel_agent.booking import parse_booking_details
    from services.booking_pdf import generate_booking_confirmation_pdf
    from services.hotel_provider.nuitee_client import NuiteeAPIError

    try:
        with console.status("Fetching your booking confirmation…"):
            d = get_booking_details(booking.booking_id)
    except (NuiteeAPIError, ValueError) as exc:
        console.print(f"[yellow]Couldn't retrieve the booking record ({exc}); "
                      "using the booking response.[/yellow]")
        d = parse_booking_details(booking.raw)

    if d.sandbox:
        console.print(Panel("TEST BOOKING: sandbox environment, no real reservation was made "
                            "and nothing was charged.", style="bold white on red"))
    if not d.is_confirmed:
        raise SystemExit(f"Booking {d.booking_id} is not confirmed (status: {d.status}). "
                         "Contact support with this booking ID.")

    na = "Not provided"
    rooms = "\n".join(f"  • {r.room_type or na} ({r.board or na}): {', '.join(r.guests) or na}"
                      for r in d.rooms) or f"  {na}"
    console.print(Panel(
        f"[bold]{d.hotel_name or na}[/bold]\n{d.hotel_address or ''}\n\n"
        f"Booking ID: [bold]{d.booking_id}[/bold] · Hotel confirmation: "
        f"{d.hotel_confirmation_code or na}\n"
        f"Check-in: {d.check_in or na} {('from ' + d.check_in_time) if d.check_in_time else ''}\n"
        f"Check-out: {d.check_out or na} {('by ' + d.check_out_time) if d.check_out_time else ''}\n"
        f"Guests: {d.guest_count if d.guest_count is not None else na} · "
        f"Rooms: {len(d.rooms) or na}\n{rooms}\n"
        f"Total: [green]{money(d.total, d.currency)}[/green] · Payment: {d.payment_status or na}\n"
        f"Cancellation: {'non-refundable' if d.refundable is False else 'refundable' if d.refundable else na}",
        title="✅ Booking confirmed", border_style="green"))

    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / f"booking_confirmation_{d.booking_id}.pdf"
    path.write_bytes(generate_booking_confirmation_pdf(d))
    console.print(f"PDF saved: [bold]{path.resolve()}[/bold]")


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    console.print(Panel(f"[bold cyan]{APP_NAME}[/bold cyan] v{APP_VERSION}\n"
                        "[dim]Gemini · LangGraph · Nuitee (LiteAPI)[/dim]",
                        border_style="cyan", padding=(1, 4)))
    initialize_client(bootstrap_credentials())

    try:
        payload = collect_trip()
        result = find_hotels(payload)
        if not result.hotels:
            raise SystemExit("No hotels matched this search.")
        booking = book(pick_and_prebook(result))
        confirm_and_save(booking, Path(args.pdf_dir))
    except Quit:
        console.print("\n[dim]Session ended.[/dim]")
        return
    console.print("\n[bold cyan]Safe travels! ✈[/bold cyan]")


if __name__ == "__main__":
    run()
