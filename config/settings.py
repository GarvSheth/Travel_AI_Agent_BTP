from pathlib import Path

DEFAULT_PROVIDER = "openrouter"  # "gemini" | "anthropic" | "openrouter"

GEMINI_MODEL_ID = "gemini-3.8-flash"
ANTHROPIC_MODEL_ID = "claude-sonnet-4-6"
# Any model slug from https://openrouter.ai/models that supports tool use
OPENROUTER_MODEL_ID = "openai/gpt-4o-mini"

# Kept for backwards compat — nodes should use get_client() from llm_client.py
MODEL_ID = ANTHROPIC_MODEL_ID

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

# ── Hotel agent ───────────────────────────────────────────────────────────────
HOTEL_DESCRIPTION_QUESTION = "What are the main features, atmosphere, and highlights of this hotel?"
HOTEL_MIN_RATING = 0.0         # minimum guest rating (0-10); 0 = no filter
HOTEL_SEARCH_RADIUS_M = 5000   # search radius in metres around the resolved location
HOTEL_SEARCH_LIMIT = 50        # max hotels fetched per search
HOTEL_TOP_N = 5                # max hotel recommendations shown to user
HOTEL_LOCATION_MAX_KM_FROM_CITY = 50    # geocoded landmark must be within this distance
HOTEL_MAX_LOCATION_CLARIFICATIONS = 3   # max times the agent asks about location

# ── Flight agent ──────────────────────────────────────────────────────────────
FLIGHT_SEARCH_LIMIT = 30       # max offers fetched per Duffel search
FLIGHT_TOP_N = 5               # max flight recommendations shown to user
FLIGHT_MAX_CLARIFICATION_ATTEMPTS = 2   # max times the agent asks about preferences
