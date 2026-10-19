"""Shared Streamlit pieces for driving the hotel LangGraph workflow and rendering results.

Used by the chat page (streamlit_app.py) and the Hotel Finder page.
"""
from __future__ import annotations

import os

import streamlit as st


@st.cache_resource(show_spinner=False)
def _init_llm(key: str) -> None:
    from services.llm_client import initialize_client
    initialize_client(key)


def _guard(call):
    """Run a workflow call; turn unexpected failures into an error outcome."""
    from agents.hotel_agent import HotelWorkflowOutcome

    try:
        _init_llm(os.getenv("GEMINI_API_KEY", ""))
        return call()
    except Exception as exc:  # the graph reports expected failures itself
        return HotelWorkflowOutcome(thread_id="", status="error", error=str(exc))


def start_hotel_workflow(request):
    """Start the LangGraph hotel workflow. Returns a HotelWorkflowOutcome."""
    from agents.hotel_agent import search_hotels
    return _guard(lambda: search_hotels(request))


def resume_hotel_workflow(thread_id: str, answer: str):
    """Answer the location question on the same graph thread."""
    from agents.hotel_agent import resume_hotel_search
    return _guard(lambda: resume_hotel_search(thread_id, answer))


def fetch_room_offers(hotel_id: str, stay):
    """Live prices for one hotel. Returns (HotelRates, None) or (None, error message)."""
    from agents.hotel_agent import HotelAgent
    from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeAuthError

    if stay is None:
        return None, "Trip dates are missing, so prices can't be checked."
    try:
        return HotelAgent().get_room_offers(hotel_id, stay), None
    except NuiteeAuthError:
        return None, "Hotel API key missing. Set NUITEE_API_KEY in .env."
    except NuiteeAPIError as exc:
        return None, f"Price check failed: {exc}"
    except ValueError as exc:  # invalid dates / guests
        return None, str(exc)


def _money(amount: float, currency: str) -> str:
    return f"{amount:,.0f} {currency}".strip()


def _yes_no(flag) -> str:
    return {True: "✓", False: "✗"}.get(flag, "?")


def render_room_offers(hotel_name: str, rates, on_prebook=None, key: str = "rates") -> None:
    """Room offers for a selected hotel, cheapest first, each with a Prebook button.

    ``on_prebook(offer)`` runs once when that offer's Prebook button is clicked.
    """
    stay = rates.stay
    nights = stay.nights
    st.markdown(f"#### 💰 Prices at {hotel_name}")
    st.caption(f"{stay.check_in} → {stay.check_out} · {nights} night(s) · "
               f"{stay.adults} adult(s) · {stay.rooms} room(s)")
    if not rates.offers:
        st.info("No rooms available for these dates.")
        return

    widths = [3, 2, 1, 2, 2, 2]
    for col, label in zip(st.columns(widths), ["Room", "Board", "Refundable",
                                               f"Total ({nights} night(s))", "≈ per night", ""]):
        col.markdown(f"**{label}**")
    for i, o in enumerate(rates.offers):
        c = st.columns(widths, vertical_alignment="center")
        c[0].write(o.room_name)
        c[1].write(o.board or "—")
        c[2].write(_yes_no(o.refundable))
        c[3].write(_money(o.total, o.currency))
        c[4].write(_money(o.total / nights, o.currency) if nights else "—")
        if on_prebook is not None and c[5].button("Prebook", key=f"{key}_prebook_{i}",
                                                  width="stretch"):
            on_prebook(o)
            st.rerun()
    if rates.sandbox:
        st.caption("⚠️ Sandbox prices: test data from the Nuitee sandbox key, not bookable.")


def prebook_room(offer, hotel_name: str):
    """Prebook one offer. Returns (PrebookResult, None) or (None, error message)."""
    from agents.hotel_agent import HotelAgent
    from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeAuthError

    try:
        return HotelAgent().prebook(offer.offer_id, hotel_name), None
    except NuiteeAuthError:
        return None, "Hotel API key missing. Set NUITEE_API_KEY in .env."
    except NuiteeAPIError as exc:
        return None, f"Prebook failed: {exc} The offer may have expired; select the hotel again."
    except ValueError as exc:
        return None, str(exc)


def book_room(prebook, guest):
    """Final booking. Returns (BookingResult, None) or (None, error message)."""
    from agents.hotel_agent import HotelAgent
    from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeAuthError

    try:
        return HotelAgent().book(prebook, guest), None
    except NuiteeAuthError:
        return None, "Hotel API key missing. Set NUITEE_API_KEY in .env."
    except NuiteeAPIError as exc:
        return None, f"Booking failed: {exc}"
    except ValueError as exc:  # missing guest details
        return None, str(exc)


def render_prebook(prebook, booked: bool, on_book=None) -> None:
    """The held room, plus the guest form and Book button until it is booked.

    ``on_book(prebook, guest)`` runs once with valid guest details.
    """
    from agents.hotel_agent import GuestDetails

    st.markdown(f"#### 🛎️ Room held at {prebook.hotel_name}")
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("Room", prebook.room_name)
        b.metric("Total", _money(prebook.price, prebook.currency))
        c.metric("Refundable", _yes_no(prebook.refundable))
        st.caption(f"{prebook.check_in} → {prebook.check_out} · "
                   f"{prebook.board or 'board unknown'} · {prebook.rooms} room(s) · "
                   f"prebook ID `{prebook.prebook_id}`")
        if prebook.price_changed:
            st.warning("The price changed since the search; the total above is the new price.")
        if prebook.terms_changed:
            st.warning("The cancellation policy or meal plan changed since the search.")

        if booked:
            st.success("Booked. See the confirmation below.")
            return

        with st.form(f"book_{prebook.prebook_id}"):
            st.markdown("**Guest details** (used for every room)")
            f1, f2 = st.columns(2)
            first = f1.text_input("First name")
            last = f2.text_input("Last name")
            f3, f4 = st.columns(2)
            email = f3.text_input("Email")
            phone = f4.text_input("Phone", placeholder="+91…")
            clicked = st.form_submit_button("Book", type="primary", width="stretch")
        if prebook.sandbox:
            st.caption("⚠️ Sandbox key: the booking is simulated and nothing is charged.")
        else:
            st.caption("Booking charges the card stored on your Nuitee account.")

    if clicked:
        guest = GuestDetails(first, last, email, phone)
        missing = [n for n, v in (("first name", first), ("last name", last),
                                  ("email", email), ("phone", phone)) if not v.strip()]
        if missing:
            st.error("Please fill in: " + ", ".join(missing) + ".")
        elif "@" not in email:
            st.error("Please enter a valid email address.")
        elif on_book is not None:
            on_book(prebook, guest)
            st.rerun()


def load_booking_details(booking):
    """Booking details for the confirmation: GET /bookings/{id} (source of truth).

    Falls back to the saved /rates/book response if retrieval fails. Returns
    (BookingDetails | None, note | None).
    """
    from agents.hotel_agent import get_booking_details
    from agents.hotel_agent.booking import parse_booking_details
    from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeAuthError

    try:
        return get_booking_details(booking.booking_id), None
    except (NuiteeAPIError, NuiteeAuthError, ValueError) as exc:
        note = f"Couldn't retrieve the booking record ({exc}); showing the booking response."
    try:
        return parse_booking_details(booking.raw), note
    except NuiteeAPIError:
        return None, note


def _or_na(value) -> str:
    return "Not provided" if value in (None, "", []) else str(value)


def render_booking_confirmation(details, pdf_bytes: bytes | None, note: str | None = None) -> None:
    """Booking confirmation section with a PDF download (only when CONFIRMED)."""
    if details.sandbox:
        st.warning("**TEST BOOKING:** sandbox environment, no real reservation was made "
                   "and nothing was charged.")
    if note:
        st.caption(note)

    if not details.is_confirmed:
        st.error(f"Booking **{details.booking_id}** is not confirmed (status: "
                 f"{details.status}). Please contact support with this booking ID.")
        return

    st.success(f"### ✅ Booking confirmed\n{_or_na(details.hotel_name)}")
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("Booking ID", details.booking_id)
        b.metric("Hotel confirmation", _or_na(details.hotel_confirmation_code))
        c.metric("Total", _money(details.total, details.currency or "")
                 if details.total is not None else "Not provided")

        d1, d2, d3 = st.columns(3)
        d1.metric("Check-in", _or_na(details.check_in),
                  details.check_in_time and f"from {details.check_in_time}", delta_color="off")
        d2.metric("Check-out", _or_na(details.check_out),
                  details.check_out_time and f"by {details.check_out_time}", delta_color="off")
        d3.metric("Payment", _or_na(details.payment_status).capitalize())

        st.markdown(f"**Status:** {details.status} · **Guests:** {_or_na(details.guest_count)} · "
                    f"**Rooms:** {len(details.rooms) or 'Not provided'}")
        if details.hotel_address:
            st.caption(f"📍 {details.hotel_address}"
                       + (f" · ☎ {details.hotel_phone}" if details.hotel_phone else ""))

        if details.rooms:
            st.dataframe([{
                "Room type": _or_na(r.room_type), "Board": _or_na(r.board),
                "Adults": _or_na(r.adults), "Guests": _or_na(", ".join(r.guests)),
            } for r in details.rooms], hide_index=True, width="stretch")

        st.markdown("**Cancellation policy**")
        if details.refundable is False:
            st.write("Non-refundable: the booking cannot be cancelled for a refund.")
        elif details.refundable is True:
            st.write("Refundable, subject to the rules below.")
        if details.last_free_cancellation:
            st.write(f"Free cancellation until {details.last_free_cancellation}.")
        for rule in details.cancellation_rules:
            st.write(f"• {rule}")
        if details.refundable is None and not details.cancellation_rules:
            st.write("Not provided")

        if details.remarks:
            with st.expander("Important hotel remarks", expanded=False):
                for r in details.remarks:
                    st.write(r)

        if pdf_bytes:
            st.download_button(
                "⬇️ Download confirmation (PDF)", data=pdf_bytes,
                file_name=f"booking_confirmation_{details.booking_id}.pdf",
                mime="application/pdf", key=f"pdf_{details.booking_id}",
                type="primary", width="stretch",
            )
        st.caption("Booking summary from this app, not a hotel invoice or supplier voucher.")


def _fmt(value, suffix: str = "") -> str:
    return f"{value:g}{suffix}" if isinstance(value, (int, float)) else "—"


def _hotel_card(hotel, selected_key: str, reference_label: str, on_select=None) -> None:
    selected = st.session_state.get(selected_key) == hotel.hotel_id
    with st.container(border=True):
        img_col, info_col = st.columns([2, 3], gap="medium")

        with img_col:
            if hotel.image_url:
                st.image(hotel.image_url, width="stretch")
            else:
                st.caption("No image available")

        with info_col:
            title = f"#{hotel.rank} · {hotel.name}"
            st.subheader(f"✅ {title}" if selected else title)

            place = hotel.address or hotel.city or "Address unavailable"
            if hotel.distance_km is not None:
                place += f" · {hotel.distance_km:.1f} km from {reference_label}"
            st.caption(f"📍 {place}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Match score", f"{hotel.match_score}/100")
            m2.metric("Stars", "★" * int(hotel.stars) if hotel.stars else "—")
            reviews = f" ({hotel.review_count:,} reviews)" if hotel.review_count else ""
            m3.metric("Guest rating", _fmt(hotel.guest_rating, "/10") + reviews)

            st.markdown(f"**Why it matches:** {hotel.explanation or '—'}")

            if hotel.description:
                with st.expander("About this hotel", expanded=False):
                    st.write(hotel.description)
            else:
                st.info(hotel.description_error or "Description unavailable for this hotel.")

            if selected:
                st.success("Selected")
            elif st.button("Select this hotel", key=f"{selected_key}_{hotel.hotel_id}",
                           width="stretch"):
                st.session_state[selected_key] = hotel.hotel_id
                if on_select is not None:
                    on_select(hotel)
                st.rerun()


def render_hotel_results(result, selected_key: str = "selected_hotel_id", on_select=None) -> None:
    """Render warnings, hotel cards and the current selection.

    ``on_select(hotel)`` runs once when a hotel's Select button is clicked.
    """
    for warning in result.warnings:
        st.warning(warning)

    if result.candidates_found == 0:
        st.info("No hotels found for these filters. Try widening the star rating or "
                "lowering the minimum guest rating.")
        return
    if not result.hotels:
        st.info(f"Found {result.candidates_found} hotels, but none met your must-haves.")
        return

    ref = result.location.label.split(",")[0] if result.location else ""
    near = f" near {ref}" if ref else ""
    st.markdown(f"#### Top {len(result.hotels)} of {result.candidates_found} hotels{near}")
    if len(result.hotels) < 5:
        st.caption(f"Only {len(result.hotels)} hotel(s) matched your preferences.")
    for hotel in result.hotels:
        _hotel_card(hotel, selected_key, ref or "your chosen location", on_select)
    st.caption("Descriptions come from Nuitee's hotel information. Select a hotel to "
               "check live room prices for your dates.")

    chosen = next((h for h in result.hotels
                   if h.hotel_id == st.session_state.get(selected_key)), None)
    if chosen:
        st.success(f"You selected **{chosen.name}**.")
