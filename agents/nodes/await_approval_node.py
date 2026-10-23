from langgraph.types import interrupt


def await_approval_node(state: dict) -> dict:
    answer = interrupt({
        "prompt": "Approve plan? [y = yes / n = regenerate with feedback]: ",
        "message": "\n[bold]Does this itinerary look good?[/bold]",
    })
    approved = answer.strip().lower() in ("y", "yes", "1", "ok", "sure", "looks good")

    messages = list(state.get("messages", []))
    if not approved:
        messages.append({"role": "user", "content": answer})

    return {"plan_approved": approved, "messages": messages}
