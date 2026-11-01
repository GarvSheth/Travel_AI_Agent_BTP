"""Debug terminal utility.

Displays the final orchestration input in a separate terminal window when
SHOW_ORCHESTRATOR_INPUT_TERMINAL=true is set in the environment.

Falls back to a log file + inline print when no graphical terminal is available
(headless servers, SSH sessions, containers).

Launch logic is intentionally kept in one small module so it can be mocked in tests.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Guard: prevents duplicate windows if LangGraph retries node execution
_SESSION_SHOWN = False

# Keys whose values are always redacted in the displayed copy
_REDACT_SUBSTRINGS = {
    "api_key", "api_secret", "password", "token", "secret",
    "card_number", "cvv", "authorization", "cookie", "credential",
}


# ── Redaction ─────────────────────────────────────────────────────────────────

def redact_for_display(obj: object) -> object:
    """Deep-copy and redact sensitive keys. The original object is never modified."""
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if any(s in k.lower() for s in _REDACT_SUBSTRINGS) else redact_for_display(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact_for_display(item) for item in obj]
    return obj


# ── Formatting ────────────────────────────────────────────────────────────────

def format_payload(payload: dict, invocation_id: str) -> str:
    safe = redact_for_display(copy.deepcopy(payload))
    lines = [
        "=" * 60,
        "         FINAL INPUT TO ORCHESTRATOR",
        "=" * 60,
        f"Invocation ID : {invocation_id}",
        "",
        json.dumps(safe, indent=2, ensure_ascii=False, default=str),
        "",
        "=" * 60,
        "Press Enter to close this window.",
    ]
    return "\n".join(lines)


# ── Terminal detection ────────────────────────────────────────────────────────

def _find_terminal() -> tuple[str, list[str]] | None:
    """Return (binary, args_template) for the first available terminal emulator."""
    candidates = [
        ("gnome-terminal", ["gnome-terminal", "--", "bash", "-c", "{script}"]),
        ("xterm",          ["xterm",          "-e",   "bash {script}"]),
        ("konsole",        ["konsole",        "-e",   "bash {script}"]),
        ("xfce4-terminal", ["xfce4-terminal", "-e",   "bash {script}"]),
        ("lxterminal",     ["lxterminal",     "-e",   "bash {script}"]),
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", "bash {script}"]),
    ]
    for binary, args in candidates:
        if shutil.which(binary):
            return binary, args
    return None


def _write_display_script(content: str) -> str:
    """Write a temporary shell script that prints content and waits for Enter."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="travel_agent_debug_"
    )
    # Use a heredoc so the content is never interpreted as shell commands
    f.write("#!/bin/bash\n")
    f.write("cat << 'ENDOFPAYLOAD'\n")
    f.write(content)
    f.write("\nENDOFPAYLOAD\n")
    f.write('read -rp "" _\n')  # wait for Enter
    f.flush()
    f.close()
    os.chmod(f.name, 0o700)
    return f.name


# ── Public API ────────────────────────────────────────────────────────────────

def show_orchestrator_input(payload: dict, invocation_id: str) -> None:
    """Display the orchestration payload. Behaviour controlled by env vars.

    SHOW_ORCHESTRATOR_INPUT_TERMINAL=true  →  open a new terminal window
    SHOW_ORCHESTRATOR_INPUT_TERMINAL=false →  do nothing (default)

    Falls back to a log file + inline print when no graphical terminal is found.
    """
    global _SESSION_SHOWN
    if _SESSION_SHOWN:
        return  # already displayed once this process run

    enabled = os.getenv("SHOW_ORCHESTRATOR_INPUT_TERMINAL", "false").strip().lower()
    if enabled not in ("true", "1", "yes"):
        return

    content = format_payload(payload, invocation_id)

    # Attempt graphical terminal only when $DISPLAY is set (not headless)
    terminal = _find_terminal()
    has_display = bool(os.getenv("DISPLAY", "").strip())

    if terminal and has_display:
        binary, arg_template = terminal
        script_path = _write_display_script(content)
        # Expand {script} placeholder in the arg template
        args = [a.replace("{script}", script_path) for a in arg_template]
        try:
            subprocess.Popen(args, start_new_session=True, close_fds=True)
            _SESSION_SHOWN = True
            return
        except OSError:
            pass  # terminal launch failed — fall through to file fallback

    # Fallback: write a JSON log file and print inline
    log_dir = Path.home() / ".travel-agent" / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"orchestrator_input_{invocation_id[:8]}.json"
    log_file.write_text(content, encoding="utf-8")

    print("\n" + content)
    print(f"\n[Debug] Payload saved to: {log_file}\n")
    _SESSION_SHOWN = True


def reset_session_guard() -> None:
    """Reset the deduplication guard. Useful in tests."""
    global _SESSION_SHOWN
    _SESSION_SHOWN = False
