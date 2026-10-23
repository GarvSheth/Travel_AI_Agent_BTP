from langgraph.types import interrupt


def await_decision_node(state: dict) -> dict:
    idx = state["current_booking_index"]
    item = state["bookable_items"][idx]
    scored = state["search_results"].get(str(idx), [])

    if not scored:
        # No options passed filters — auto-skip
        return {"item_decision": "skip"}

    count = min(len(scored), 5)
    answer = interrupt({
        "prompt": f"Choice [1–{count}/s/m]: ",
        "message": (
            f"\n[bold]Book this {item['type']}?[/bold]  "
            f"Enter [cyan]1–{count}[/cyan] to book · [cyan]s[/cyan] to skip · [cyan]m[/cyan] for more options"
        ),
    })
    answer = answer.strip().lower()

    new_results = dict(state["search_results"])

    if answer.isdigit():
        choice = int(answer) - 1
        if 0 <= choice < len(scored):
            selected = scored[choice]
            # Store raw option dict as the selected item
            opt = selected["option"] if isinstance(selected, dict) else selected.option
            new_results[f"{idx}_selected"] = opt if isinstance(opt, dict) else opt.model_dump()
        else:
            # Out of range — pick first
            opt = scored[0]["option"] if isinstance(scored[0], dict) else scored[0].option
            new_results[f"{idx}_selected"] = opt if isinstance(opt, dict) else opt.model_dump()
        return {"item_decision": "book", "search_results": new_results}

    if answer == "m":
        return {"item_decision": "show_more"}

    return {"item_decision": "skip"}
