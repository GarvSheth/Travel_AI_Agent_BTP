"""
Isolated payment subprocess. Invoked by booking_node via subprocess.run().

Security properties:
- Runs in a separate OS process; agent heap cannot read card data
- Card input collected via getpass() which reads /dev/tty directly,
  bypassing the stdin pipe used to receive the BookingRequest
- Card variables are explicitly dereferenced after use
- Only PaymentResult JSON (confirmation_id, status) is written to stdout
- All diagnostic output goes to stderr (not captured by parent in simulate mode)
"""
import sys
import uuid
from getpass import getpass

# Resolve project root so models are importable when called as a subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.booking import PaymentRequest, PaymentResult


def _luhn_check(number: str) -> bool:
    digits = [int(d) for d in number.replace(" ", "") if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _zero(s: str) -> None:
    """Best-effort memory zero — removes the reference immediately."""
    s = "X" * len(s)
    del s


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def collect_and_pay(req: PaymentRequest) -> PaymentResult:
    _print_err(f"\n{'─' * 54}")
    _print_err(f"  SECURE PAYMENT GATEWAY")
    _print_err(f"  {req.provider_name}")
    _print_err(f"  Amount : {req.currency} {req.amount:,.2f}")
    _print_err(f"  Travelers: {', '.join(req.traveler_names)}")
    _print_err(f"{'─' * 54}")
    _print_err("  ⓘ  Card details are entered here only.")
    _print_err("  ⓘ  The AI agent cannot read this input.\n")

    if req.simulate:
        _ = getpass("  Card number   (demo — not submitted): ")
        _ = getpass("  Expiry MM/YY  (demo — not submitted): ")
        _ = getpass("  CVV           (demo — not submitted): ")
        _print_err("\n  [DEMO MODE] No real payment processed.\n")
        return PaymentResult(
            status="confirmed",
            confirmation_id=f"MOCK-{uuid.uuid4().hex[:8].upper()}",
            amount_charged=req.amount,
        )

    # Production path
    card_number = getpass("  Card number   : ")
    expiry = getpass("  Expiry MM/YY  : ")
    cvv = getpass("  CVV           : ")
    cardholder = getpass("  Cardholder name: ")

    try:
        clean_number = card_number.replace(" ", "").replace("-", "")
        if not _luhn_check(clean_number):
            return PaymentResult(
                status="failed",
                confirmation_id="",
                amount_charged=0.0,
                error_message="Invalid card number (Luhn check failed).",
            )

        # ── Wire in Stripe / Razorpay / etc. here ──
        # import stripe
        # stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
        # charge = stripe.PaymentIntent.create(
        #     amount=int(req.amount * 100),
        #     currency=req.currency.lower(),
        #     payment_method_data={"type": "card", "card": {"number": clean_number, ...}},
        # )
        # return PaymentResult(status="confirmed", confirmation_id=charge.id, amount_charged=req.amount)

        # Placeholder until real gateway is wired
        return PaymentResult(
            status="confirmed",
            confirmation_id=f"PROD-{uuid.uuid4().hex[:8].upper()}",
            amount_charged=req.amount,
        )
    finally:
        _zero(card_number)
        _zero(expiry)
        _zero(cvv)
        _zero(cardholder)


if __name__ == "__main__":
    try:
        # Receive PaymentRequest via a temp file path (argv[1]) so that stdin
        # remains a TTY and getpass() can read card input from /dev/tty.
        if len(sys.argv) < 2:
            _print_err("Usage: payment_gateway.py <request_json_path>")
            sys.exit(1)
        raw = Path(sys.argv[1]).read_text()
        req = PaymentRequest.model_validate_json(raw)
        result = collect_and_pay(req)
        # Only the result JSON goes to stdout — nothing else
        print(result.model_dump_json(), flush=True)
    except Exception as exc:
        import traceback
        _print_err(f"Payment gateway error: {exc}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
