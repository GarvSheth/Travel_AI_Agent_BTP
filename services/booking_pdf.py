"""Booking confirmation PDF (ReportLab), built only from a parsed BookingDetails.

No API calls here. Missing values are shown as "Not provided"; nothing is
invented. The document is a booking summary from this app, not a hotel invoice
or a supplier-issued voucher, and says so.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config.settings import APP_NAME

BRAND = colors.HexColor("#1F4E79")
MUTED = colors.HexColor("#5F6B7A")
TEST_RED = colors.HexColor("#B42318")
NA = "Not provided"


def _p(text, style) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _money(amount, currency) -> str:
    if amount is None:
        return NA
    return f"{amount:,.2f} {currency or ''}".strip()


def _nights(check_in, check_out) -> int | None:
    try:
        return (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    except (TypeError, ValueError):
        return None


def _cancellation_text(d) -> list[str]:
    lines = []
    if d.refundable is False:
        lines.append("Non-refundable: the booking cannot be cancelled for a refund.")
    elif d.refundable is True:
        lines.append("Refundable, subject to the rules below.")
    if d.last_free_cancellation:
        lines.append(f"Free cancellation until {d.last_free_cancellation}.")
    lines += d.cancellation_rules
    return lines or [NA]


def generate_booking_confirmation_pdf(booking_details) -> bytes:
    """Render a BookingDetails as an A4 PDF and return its bytes."""
    d = booking_details
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Booking confirmation {d.booking_id}", author=APP_NAME)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=11, textColor=MUTED)
    label = ParagraphStyle("label", parent=body, fontName="Helvetica-Bold", textColor=MUTED)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=BRAND, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=BRAND,
                        spaceBefore=10, spaceAfter=4)
    brand = ParagraphStyle("brand", parent=body, fontName="Helvetica-Bold", fontSize=11,
                           textColor=BRAND, alignment=TA_CENTER)
    banner = ParagraphStyle("banner", parent=body, fontName="Helvetica-Bold", fontSize=11,
                            textColor=colors.white, alignment=TA_CENTER)

    def kv_table(rows):
        t = Table([[_p(k, label), _p(v if v not in (None, "") else NA, body)] for k, v in rows],
                  colWidths=[48 * mm, None])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#D9DEE5")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    story = [_p(APP_NAME.upper(), brand), Spacer(1, 4),
             _p("Booking Confirmation", h1)]

    if d.sandbox:
        t = Table([[_p("TEST BOOKING - sandbox environment. No real reservation was made "
                       "and nothing was charged.", banner)]], colWidths=[None])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEST_RED),
                               ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story += [Spacer(1, 4), t]

    story += [Spacer(1, 8), kv_table([
        ("Booking ID", d.booking_id),
        ("Status", d.status),
        ("Hotel confirmation code", d.hotel_confirmation_code),
        ("Booked on", d.created_at),
    ])]

    nights = _nights(d.check_in, d.check_out)
    story += [_p("Hotel", h2), kv_table([
        ("Hotel", d.hotel_name),
        ("Address", d.hotel_address),
        ("Phone", d.hotel_phone),
    ])]
    story += [_p("Stay", h2), kv_table([
        ("Check-in", " ".join(filter(None, (d.check_in, f"from {d.check_in_time}" if d.check_in_time else None))) or None),
        ("Check-out", " ".join(filter(None, (d.check_out, f"by {d.check_out_time}" if d.check_out_time else None))) or None),
        ("Nights", nights),
        ("Guests", d.guest_count),
        ("Rooms", len(d.rooms) if d.rooms else None),
    ])]

    story.append(_p("Rooms and guests", h2))
    if d.rooms:
        head = [_p(h, label) for h in ("#", "Room type", "Board", "Adults", "Guest names")]
        rows = [[_p(i, body), _p(r.room_type or NA, body), _p(r.board or NA, body),
                 _p(r.adults if r.adults is not None else NA, body),
                 _p(", ".join(r.guests) or NA, body)]
                for i, r in enumerate(d.rooms, start=1)]
        t = Table([head] + rows, colWidths=[8 * mm, 50 * mm, 32 * mm, 16 * mm, None])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF3F8")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9DEE5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
    else:
        story.append(_p(NA, body))
    story += [Spacer(1, 4), kv_table([
        ("Booking holder", d.holder_name),
        ("Email", d.holder_email),
        ("Phone", d.holder_phone),
    ])]

    story += [_p("Payment", h2), kv_table([
        ("Total amount", _money(d.total, d.currency)),
        ("Payment status", d.payment_status),
    ])]

    story.append(_p("Cancellation policy", h2))
    story += [_p(line, body) for line in _cancellation_text(d)]

    story.append(_p("Important hotel remarks and instructions", h2))
    story += [_p(r, body) for r in d.remarks] or [_p(NA, body)]

    story += [Spacer(1, 12), _p(
        f"This document is a booking summary generated by {APP_NAME} from the booking "
        "record. It is not a hotel invoice or a supplier-issued voucher. Please keep the "
        "booking ID for any enquiries.", small)]

    doc.build(story)
    return buf.getvalue()
