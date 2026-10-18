import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from rich.console import Console

from agents.graph import build_graph
from agents.state import INITIAL_STATE
from config.settings import APP_NAME, APP_VERSION
from services.credential_manager import CredentialManager
from services.llm_client import initialize_client
from ui.console import print_welcome

console = Console()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="travel-agent",
        description=f"{APP_NAME} v{APP_VERSION} — AI-powered CLI travel planner",
    )
    return parser.parse_args()


def bootstrap_credentials(provider: str) -> CredentialManager:
    load_dotenv()
    cm = CredentialManager(provider=provider)
    missing = cm.load_from_environment()
    if missing:
        names = "\n".join(f"  - {name}" for name in missing)
        raise SystemExit(
            "Backend credentials are missing. Configure these environment variables "
            "in .env or your deployment secret manager:\n"
            f"{names}"
        )
    return cm


def configure_storage() -> None:
    if not os.getenv("MONGODB_URI", "").strip():
        console.print(
            "[yellow]⚠  MONGODB_URI not set — using in-memory storage "
            "(data won't persist)[/yellow]\n"
        )


def _collect_initial_prompt() -> str:
    """Ask the user for their travel request before starting the graph."""
    console.print("[bold]Tell me about your trip[/bold] (or press Enter to start a conversation):\n")
    try:
        raw = console.input("[bold yellow]Your request: [/bold yellow]").strip()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nSession ended.")
    return raw


def run() -> None:
    parse_args()

    print_welcome()
    console.print("[dim]Provider: Gemini (Google)[/dim]\n")

    cm = bootstrap_credentials("gemini")
    configure_storage()
    api_key = cm.get_credential("GEMINI_API_KEY")
    initialize_client(api_key)

    # ── Collect initial prompt and run input pipeline ──────────────────────────
    raw_prompt = _collect_initial_prompt()

    state = dict(INITIAL_STATE)
    state["provider"] = "gemini"
    state["profile_id"] = profile_id()

    thread_id = str(uuid.uuid4())
    state["correlation_id"] = thread_id

    if raw_prompt:
        from workflows.input_pipeline import (
            build_input_graph, orchestrator_input_to_state, start_conversation, submit_answer,
        )
        input_graph = build_input_graph()
        input_thread = f"input-{thread_id}"
        try:
            turn = start_conversation(input_graph, input_thread, raw_prompt)
            while not turn.complete:
                console.print(f"\n[bold cyan]Agent:[/bold cyan] {turn.question}")
                answer = console.input("[bold yellow]You: [/bold yellow]").strip()
                turn = submit_answer(input_graph, input_thread, answer)
        except ValueError as exc:
            console.print(f"[red]Input error: {exc}[/red]")
            return
        except (KeyboardInterrupt, EOFError):
            raise SystemExit("\nSession ended.")

        oi = turn.payload
        from services.debug_terminal import show_orchestrator_input
        show_orchestrator_input(oi.model_dump(), oi.invocation_id)
        state.update(orchestrator_input_to_state(oi))
        console.print(
            f"[dim]Trip: {oi.trip_extraction.origin} → {oi.trip_extraction.destination}, "
            f"{oi.trip_extraction.departure_date} – {oi.trip_extraction.return_date}[/dim]\n"
        )
    else:
        # No upfront prompt — full conversational flow
        state["messages"] = []

    # ── Build and run graph ────────────────────────────────────────────────────
    graph = build_graph(use_sqlite=True)
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Session: {thread_id}[/dim]\n")
    console.print("[bold green]Planning your trip…[/bold green]\n")

    result = graph.invoke(state, config)

    while True:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            break

        interrupt_val = interrupts[0].value
        if isinstance(interrupt_val, dict):
            msg = interrupt_val.get("message")
            if msg:
                console.print(msg)
            raw_prompt_str = interrupt_val.get("prompt", "You: ")
        else:
            raw_prompt_str = str(interrupt_val)
        input_prompt = (
            raw_prompt_str
            if raw_prompt_str.endswith((" ", ": "))
            else raw_prompt_str + " "
        )

        try:
            user_input = console.input(
                f"[bold yellow]{input_prompt}[/bold yellow]"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        result = graph.invoke(Command(resume=user_input), config)

    console.print("\n[bold cyan]Safe travels! ✈[/bold cyan]")


if __name__ == "__main__":
    run()
