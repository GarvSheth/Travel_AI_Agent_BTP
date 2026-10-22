"""Streamlit conversational frontend for the Travel AI Agent.

One chat session drives two LangGraph workflows in sequence:
  1. input pipeline (workflows/input_pipeline) collects the trip details;
  2. its output is mapped straight onto the hotel workflow (agents/hotel_agent),
     whose accommodation-location question, and finally its hotel
     recommendations, appear as further messages in the same chat.

Only the chat messages and the hotel graph's outcome (thread id for resuming)
are kept in session state; the input pipeline's payload is not stored or shown.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

api_key = os.getenv("GEMINI_API_KEY", "")
_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
if _model:
    os.environ["GEMINI_MODEL"] = _model


@st.cache_resource(show_spinner=False)
def _get_llm(key: str):
    from services.llm_client import initialize_client, get_client
    initialize_client(key)
    return get_client()


@st.cache_resource(show_spinner=False)
def _get_input_graph():
    # Cached per process so the MemorySaver checkpoints survive Streamlit reruns.
    from workflows.input_pipeline import build_input_graph
    return build_input_graph()


def _init_state():
    defaults: dict = {
        "messages": [],             # chat log: {"role", "content"} | {"role", "kind": "hotels"}
                                    # | {"role", "kind": "rates", "hotel_id", "hotel_name"}
                                    # | {"role", "kind": "prebook" | "booking", "prebook_id"}
        "thread_id": uuid.uuid4().hex,  # input-pipeline graph thread
        "done": False,              # input pipeline finished
        "hotel_outcome": None,      # latest HotelWorkflowOutcome (holds the hotel graph thread id)
        "hotel_selected": None,
        "hotel_rates": {},          # hotel_id -> HotelRates (live prices, fetched once)
        "prebooks": {},             # prebook_id -> PrebookResult (held room)
        "bookings": {},             # prebook_id -> BookingResult (incl. full /rates/book response)
        "booking_details": {},      # booking_id -> (BookingDetails, note) from GET /bookings/{id}
        "booking_pdfs": {},         # booking_id -> confirmation PDF bytes
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _say(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})


def _hotel_waiting() -> bool:
    outcome = st.session_state.hotel_outcome
    return outcome is not None and outcome.needs_input


def _apply_hotel_outcome(outcome) -> None:
    """Turn the hotel graph's outcome into the next chat message."""
    st.session_state.hotel_outcome = outcome
    if outcome.needs_input:
        _say(outcome.question)
    elif outcome.status == "error":
        _say(f"Sorry, I couldn't find hotels: {outcome.error}")
    else:
        st.session_state.messages.append({"role": "assistant", "kind": "hotels"})


def _start_hotels(payload) -> None:
    """Input pipeline output -> hotel workflow, directly (nothing stored in between)."""
    from agents.hotel_agent import hotel_request_from_trip
    from ui.hotel_view import start_hotel_workflow

    try:
        request = hotel_request_from_trip(payload)
    except ValueError as exc:
        _say(str(exc))
        return
    _apply_hotel_outcome(start_hotel_workflow(request))


def _handle_user_message(text: str) -> None:
    from workflows.input_pipeline import handle_user_message

    turn_index = sum(1 for m in st.session_state.messages if m["role"] == "user")
    result = handle_user_message(_get_input_graph(), st.session_state.thread_id, text, turn_index)

    st.session_state.messages.append({"role": "user", "content": text})
    if not result.complete:
        _say(result.question)
    elif not st.session_state.done:
        st.session_state.done = True
        _start_hotels(result.payload)


def _on_hotel_selected(hotel) -> None:
    """Select -> live room prices for the trip's dates, added as a chat message."""
    from ui.hotel_view import fetch_room_offers

    cache = st.session_state.hotel_rates
    if hotel.hotel_id not in cache:
        stay = st.session_state.hotel_outcome.result.stay
        with st.spinner(f"Checking live prices for {hotel.name}…"):
            rates, error = fetch_room_offers(hotel.hotel_id, stay)
        if error:
            _say(f"Sorry, I couldn't get prices for {hotel.name}: {error}")
            return
        cache[hotel.hotel_id] = rates
    st.session_state.messages.append({"role": "assistant", "kind": "rates",
                                      "hotel_id": hotel.hotel_id, "hotel_name": hotel.name})


def _on_prebook(offer, hotel_name: str) -> None:
    """Prebook -> the held room appears at the end of the chat with a Book button."""
    from ui.hotel_view import prebook_room

    with st.spinner(f"Holding {offer.room_name} at {hotel_name}…"):
        prebook, error = prebook_room(offer, hotel_name)
    if error:
        _say(f"Sorry, I couldn't prebook {offer.room_name}: {error}")
        return
    st.session_state.prebooks[prebook.prebook_id] = prebook
    st.session_state.messages.append({"role": "assistant", "kind": "prebook",
                                      "prebook_id": prebook.prebook_id})


def _on_book(prebook, guest) -> None:
    """Book -> final booking via /rates/book; confirmation added to the chat."""
    from ui.hotel_view import book_room

    if prebook.prebook_id in st.session_state.bookings:     # never book twice
        return
    with st.spinner(f"Booking {prebook.room_name} at {prebook.hotel_name}…"):
        booking, error = book_room(prebook, guest)
    if error:
        _say(f"Sorry, the booking didn't go through: {error}")
        return
    st.session_state.bookings[prebook.prebook_id] = booking
    _load_confirmation(booking)
    st.session_state.messages.append({"role": "assistant", "kind": "booking",
                                      "prebook_id": prebook.prebook_id})


def _load_confirmation(booking) -> None:
    """Retrieve the booking record and build its PDF once; cached across reruns."""
    from services.booking_pdf import generate_booking_confirmation_pdf
    from ui.hotel_view import load_booking_details

    if booking.booking_id in st.session_state.booking_details:
        return
    with st.spinner("Fetching your booking confirmation…"):
        details, note = load_booking_details(booking)
    st.session_state.booking_details[booking.booking_id] = (details, note)
    if details is not None and details.is_confirmed:
        st.session_state.booking_pdfs[booking.booking_id] = (
            generate_booking_confirmation_pdf(details))


def _answer_hotel_question(text: str) -> None:
    """Resume the paused hotel graph on its own thread with the user's answer."""
    from ui.hotel_view import resume_hotel_workflow

    st.session_state.messages.append({"role": "user", "content": text})
    thread_id = st.session_state.hotel_outcome.thread_id
    _apply_hotel_outcome(resume_hotel_workflow(thread_id, text))


# ── Main UI ────────────────────────────────────────────────────────────────────
col_title, col_reset = st.columns([6, 1])
col_title.title("✈️ AI Travel Agent")
if col_reset.button("🔄 Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

try:
    _get_llm(api_key)
except Exception as exc:
    st.error(f"Could not initialise LLM client: {exc}")
    st.stop()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("kind") == "hotels":
            from ui.hotel_view import render_hotel_results
            render_hotel_results(st.session_state.hotel_outcome.result, "hotel_selected",
                                 on_select=_on_hotel_selected)
        elif msg.get("kind") == "rates":
            from ui.hotel_view import render_room_offers
            render_room_offers(msg["hotel_name"], st.session_state.hotel_rates[msg["hotel_id"]],
                               on_prebook=lambda o, name=msg["hotel_name"]: _on_prebook(o, name),
                               key=f"msg{i}")
        elif msg.get("kind") == "prebook":
            from ui.hotel_view import render_prebook
            pid = msg["prebook_id"]
            render_prebook(st.session_state.prebooks[pid], pid in st.session_state.bookings,
                           on_book=_on_book)
        elif msg.get("kind") == "booking":
            from ui.hotel_view import render_booking_confirmation
            booking = st.session_state.bookings[msg["prebook_id"]]
            details, note = st.session_state.booking_details.get(booking.booking_id, (None, None))
            if details is None:
                st.error(f"Booking {booking.booking_id} was made (status {booking.status}), "
                         "but its details couldn't be loaded. " + (note or ""))
            else:
                render_booking_confirmation(details, st.session_state.booking_pdfs.get(
                    booking.booking_id), note)
        else:
            st.markdown(msg["content"])

if not st.session_state.done or _hotel_waiting():
    placeholder = (
        "e.g. Plan a trip to Jaipur for 3-4 days"
        if not st.session_state.messages
        else "Type your answer…"
    )
    user_input = st.chat_input(placeholder)
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        if st.session_state.done:
            with st.spinner("Locating that area, then searching and ranking hotels…"):
                _answer_hotel_question(user_input)
        else:
            with st.spinner("Thinking…"):
                try:
                    _handle_user_message(user_input)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
        st.rerun()
