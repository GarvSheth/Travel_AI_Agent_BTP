from pathlib import Path

GEMINI_MODEL_ID = "gemini-3.5-flash-lite"  # override via GEMINI_MODEL env var

MAX_TOKENS = 4096

AMADEUS_HOSTNAME = "test"  # "test" for free dev tier, "production" for live

APP_DIR = Path.home() / ".travel-agent"
CHECKPOINT_DB = APP_DIR / "checkpoints.db"

MAX_SEARCH_RESULTS = 5
MAX_PREFERENCE_TURNS = 15

APP_NAME = "AI Travel Agent"
APP_VERSION = "1.0.0"

MONGODB_URI = ""
MONGODB_DATABASE = "travel_agent"
AUDIT_RETENTION_DAYS = 365
TRAVEL_DATA_MODE = "production"  # "production" or explicit "mock" for tests

# ── Hotel recommendation pipeline ─────────────────────────────────────────────
HOTEL_SEARCH_RADIUS_M = 5000      # around the user's preferred location
HOTEL_SEARCH_LIMIT = 1000         # candidates sent to the LLM for ranking
HOTEL_TOP_N = 5
HOTEL_MIN_RATING = 6.0            # minimum guest rating (0-10) sent to the hotel search
HOTEL_DESCRIPTION_QUESTION = (
    "Provide a concise and useful description of this hotel, covering its "
    "atmosphere, location, notable facilities, accommodation experience, and "
    "distinguishing features that would help a traveller decide whether to stay "
    "here. Only provide information supported by the available hotel information."
)
HOTEL_RATES_MAX_OFFERS = 6        # room offers shown when a hotel is selected
HOTEL_GUEST_NATIONALITY = "IN"    # ISO-2, sent with live-rate searches
# Payment for /rates/book. ACC_CREDIT_CARD charges the card stored on the Nuitee account;
# with a sandbox key the booking is simulated and nothing is charged.
HOTEL_PAYMENT_METHOD = "ACC_CREDIT_CARD"
HOTEL_MAX_LOCATION_CLARIFICATIONS = 2   # questions before falling back to the city
HOTEL_LOCATION_MAX_KM_FROM_CITY = 50    # geocode hits farther than this are treated as wrong
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
GEOCODER_USER_AGENT = "TravelAIAgent-BTP/1.0"   # required by Nominatim usage policy
