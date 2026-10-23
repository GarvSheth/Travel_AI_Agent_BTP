import argparse
import os
import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from rich.console import Console

from agents.graph import build_graph
from agents.state import INITIAL_STATE
from config.settings import APP_NAME, APP_VERSION, DEFAULT_PROVIDER
from services.credential_manager import CredentialManager
from services.llm_client import initialize_client
from services.profile_service import profile_id
from ui.console import print_welcome

console = Console()

PROVIDER_KEY_NAME = {
    "gemini":     "GEMINI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
PROVIDER_LABEL = {
    "gemini":     "Gemini (Google)",
    "anthropic":  "Anthropic (Claude)",
    "openrouter": "OpenRouter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="travel-agent",
        description=f"{APP_NAME} v{APP_VERSION} — AI-powered CLI travel planner",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "anthropic", "openrouter"],
        default=DEFAULT_PROVIDER,
        help="LLM provider to use (default: gemini — free tier available)",
    )
    return parser.parse_args()


def bootstrap_credentials(provider: str) -> CredentialManager:
    """Load backend-managed credentials without prompting in the CLI."""
    load_dotenv()
    cm = CredentialManager(provider=provider)
    missing = cm.load_from_environment()
    if missing:
        names = "\n".join(f"  - {name}" for name in missing)
        raise SystemExit(
            "Backend credentials are missing. Configure these environment variables "
            "in .env, your process environment, or your deployment secret manager:\n"
            f"{names}"
        )
    return cm


def configure_storage(cm: CredentialManager) -> None:
    """Validate the backend-provided MongoDB configuration."""
    if not cm.has_credential("MONGODB_URI"):
        raise SystemExit("MONGODB_URI must be configured by the backend environment.")


def run() -> None:
    args = parse_args()
    provider = args.provider

    print_welcome()
    console.print(f"[dim]Provider: {PROVIDER_LABEL[provider]}[/dim]\n")

    cm = bootstrap_credentials(provider)
    configure_storage(cm)

    # Initialize the LLM singleton before the graph runs
    api_key = cm.get_credential(PROVIDER_KEY_NAME[provider])
    initialize_client(provider, api_key)

    graph = build_graph(use_sqlite=True)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Session: {thread_id}[/dim]\n")
    console.print("[bold green]Let's start planning your trip![/bold green]\n")

    # Seed initial state with provider so nodes can access it if needed
    state = dict(INITIAL_STATE)
    state["provider"] = provider
    state["profile_id"] = profile_id()
    state["correlation_id"] = thread_id

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
            raw_prompt = interrupt_val.get("prompt", "You: ")
        else:
            raw_prompt = str(interrupt_val)
        input_prompt = raw_prompt if raw_prompt.endswith((" ", ": ")) else raw_prompt + " "

        try:
            user_input = console.input(f"[bold yellow]{input_prompt}[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        result = graph.invoke(Command(resume=user_input), config)

    console.print("\n[bold cyan]Safe travels! ✈[/bold cyan]")


if __name__ == "__main__":
    run()
