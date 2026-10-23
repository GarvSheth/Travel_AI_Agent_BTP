from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def _score_style(score: int) -> tuple[str, str]:
    """Return (rich_style, label) for a confidence score."""
    if score >= 85:
        return "bold green", "Excellent Match"
    if score >= 70:
        return "green", "Great Match"
    if score >= 55:
        return "yellow", "Good Match"
    if score >= 40:
        return "yellow", "Partial Match"
    return "red", "Poor Match"


def print_welcome() -> None:
    console.print(Panel(
        "[bold cyan]AI Travel Agent[/bold cyan]\n"
        "[dim]Powered by Claude · Amadeus · Secure payment isolation[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))


def print_itinerary(itinerary) -> None:
    console.print(Panel(
        f"[bold]{itinerary.trip_name}[/bold]\n"
        f"[dim]Destination:[/dim] {itinerary.destination}  "
        f"[dim]Days:[/dim] {itinerary.total_days}  "
        f"[dim]Est. Total:[/dim] [green]${itinerary.estimated_total_cost_usd:,.0f}[/green]",
        title="[bold]Your AI-Generated Itinerary[/bold]",
        border_style="green",
    ))
    for i, day in enumerate(itinerary.days, start=1):
        _print_day(day, i)


def _print_day(day, day_number: int) -> None:
    table = Table(
        title=f"Day {day_number}: [bold]{day.location}[/bold]  [dim]{day.date}[/dim]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Activity", style="bold", min_width=20)
    table.add_column("Description", max_width=48)
    table.add_column("Est. Cost ($)", justify="right", style="green", min_width=12)

    for act in day.activities:
        cost = f"{act.estimated_cost_usd:.0f}" if act.estimated_cost_usd else "—"
        table.add_row(act.name, act.description, cost)

    if day.flight:
        table.add_row(
            "[bold blue]✈  Flight[/bold blue]",
            f"{day.flight.origin} → {day.flight.destination}",
            "[dim blue]TBD[/dim blue]",
        )
    if day.hotel:
        table.add_row(
            "[bold magenta]🏨  Hotel[/bold magenta]",
            f"{day.hotel.city}  ({day.hotel.check_in} → {day.hotel.check_out})",
            "[dim magenta]TBD[/dim magenta]",
        )
    if day.notes:
        table.add_row("[dim]Note[/dim]", f"[dim]{day.notes}[/dim]", "")

    console.print(table)


def print_flight_options(scored_options: list, idx: int, prefs) -> None:
    filters = _flight_filter_summary(prefs)
    console.print(f"\n[dim]Active filters:[/dim] {filters}")

    table = Table(
        title=f"Flight Options  [dim](item {idx + 1})[/dim]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", width=3)
    table.add_column("Airline")
    table.add_column("Flight")
    table.add_column("Departure")
    table.add_column("Arrival")
    table.add_column("Stops", justify="center")
    table.add_column("Price ($)", justify="right")
    table.add_column("Match", min_width=18)

    for i, so in enumerate(scored_options[:5], start=1):
        opt = so["option"] if isinstance(so, dict) else so.option
        score = so["score"] if isinstance(so, dict) else so.score
        breakdown = so["breakdown"] if isinstance(so, dict) else so.breakdown
        style, label = _score_style(score)
        if isinstance(opt, dict):
            row = (str(i), opt.get("airline",""), opt.get("flight_number",""),
                   opt.get("departure_time",""), opt.get("arrival_time",""),
                   str(opt.get("stops",0)), f"{opt.get('price_usd',0):,.2f}",
                   f"[{style}]{score} {label}[/{style}]")
        else:
            row = (str(i), opt.airline, opt.flight_number,
                   opt.departure_time, opt.arrival_time,
                   str(opt.stops), f"{opt.price_usd:,.2f}",
                   f"[{style}]{score} {label}[/{style}]")
        table.add_row(*row)

    console.print(table)

    # Breakdown for top result
    if scored_options:
        top = scored_options[0]
        bd = top["breakdown"] if isinstance(top, dict) else top.breakdown
        parts = "  ·  ".join(f"{k} {v}/pts" for k, v in bd.items())
        console.print(f"  [dim]Score breakdown (option 1):[/dim] [dim]{parts}[/dim]")


def print_hotel_options(scored_options: list, idx: int, prefs) -> None:
    filters = _hotel_filter_summary(prefs)
    console.print(f"\n[dim]Active filters:[/dim] {filters}")

    table = Table(
        title=f"Hotel Options  [dim](item {idx + 1})[/dim]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", width=3)
    table.add_column("Hotel", min_width=22)
    table.add_column("Stars", justify="center")
    table.add_column("Rating", justify="center")
    table.add_column("/night ($)", justify="right")
    table.add_column("Total ($)", justify="right")
    table.add_column("Match", min_width=18)

    for i, so in enumerate(scored_options[:5], start=1):
        opt = so["option"] if isinstance(so, dict) else so.option
        score = so["score"] if isinstance(so, dict) else so.score
        style, label = _score_style(score)
        if isinstance(opt, dict):
            stars = "★" * opt.get("stars", 0)
            rating_str = f"{opt.get('rating','—')}★"
            if opt.get("review_count"):
                rating_str += f" ({opt['review_count']})"
            row = (str(i), opt.get("name",""), stars, rating_str,
                   f"{opt.get('price_per_night_usd',0):,.2f}",
                   f"{opt.get('total_price_usd',0):,.2f}",
                   f"[{style}]{score} {label}[/{style}]")
        else:
            stars = "★" * opt.stars
            rating_str = f"{opt.rating}★" if opt.rating else "—"
            if opt.review_count:
                rating_str += f" ({opt.review_count})"
            row = (str(i), opt.name, stars, rating_str,
                   f"{opt.price_per_night_usd:,.2f}",
                   f"{opt.total_price_usd:,.2f}",
                   f"[{style}]{score} {label}[/{style}]")
        table.add_row(*row)

    console.print(table)

    if scored_options:
        top = scored_options[0]
        bd = top["breakdown"] if isinstance(top, dict) else top.breakdown
        parts = "  ·  ".join(f"{k} {v}" for k, v in bd.items())
        console.print(f"  [dim]Score breakdown (option 1):[/dim] [dim]{parts}[/dim]")


def print_booking_confirmed(record) -> None:
    details = record["details"] if isinstance(record, dict) else record.details
    conf = record["confirmation_number"] if isinstance(record, dict) else record.confirmation_number
    itype = record["item_type"] if isinstance(record, dict) else record.item_type
    name = details.get("name") or details.get("airline", "Provider")
    console.print(Panel(
        f"[bold green]Booking Confirmed![/bold green]\n"
        f"[dim]{itype.title()}:[/dim] {name}\n"
        f"[dim]Confirmation:[/dim] [bold]{conf}[/bold]",
        border_style="green",
    ))


def print_trip_summary(booking_records: list, skipped: int, total_spent: float, currency: str) -> None:
    console.print(Panel(
        f"[bold]Trip Summary[/bold]\n\n"
        f"Bookings confirmed: [bold green]{len(booking_records)}[/bold green]\n"
        f"Items skipped:      [dim]{skipped}[/dim]\n"
        f"Total simulated spend: [bold green]{currency} {total_spent:,.2f}[/bold green]",
        title="[bold cyan]Your Trip is Planned![/bold cyan]",
        border_style="cyan",
    ))
    for rec in booking_records:
        details = rec["details"] if isinstance(rec, dict) else rec.details
        conf = rec["confirmation_number"] if isinstance(rec, dict) else rec.confirmation_number
        itype = rec["item_type"] if isinstance(rec, dict) else rec.item_type
        name = details.get("name") or details.get("airline", "Provider")
        console.print(f"  [green]✓[/green]  {itype.title():8s}  {name:30s}  [dim]{conf}[/dim]")


def _hotel_filter_summary(prefs) -> str:
    parts = []
    hp = prefs.hotel_preferences if hasattr(prefs, "hotel_preferences") else prefs
    if hasattr(hp, "star_rating_min"):
        parts.append(f"⭐{hp.star_rating_min}–{hp.star_rating_max}")
    if getattr(hp, "max_price_per_night", None):
        parts.append(f"≤{hp.max_price_per_night:.0f}/night")
    if getattr(hp, "min_rating", 0):
        parts.append(f"≥{hp.min_rating}★")
    if getattr(hp, "min_review_count", 0):
        parts.append(f"{hp.min_review_count}+ reviews")
    if getattr(hp, "preferred_location", None):
        parts.append(hp.preferred_location.replace("_", " "))
    return "  ".join(parts) if parts else "none"


def _flight_filter_summary(prefs) -> str:
    parts = []
    fp = prefs.flight_preferences if hasattr(prefs, "flight_preferences") else prefs
    if hasattr(fp, "max_stops"):
        stops_label = {0: "direct only", 1: "≤1 stop", 2: "any stops"}.get(fp.max_stops, "")
        if stops_label:
            parts.append(stops_label)
    if getattr(fp, "max_price", None):
        parts.append(f"≤{fp.max_price:.0f}")
    if getattr(fp, "cabin_class", "economy") != "economy":
        parts.append(fp.cabin_class)
    if getattr(fp, "preferred_departure_time", "any") != "any":
        parts.append(fp.preferred_departure_time)
    return "  ".join(parts) if parts else "none"
