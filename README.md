# AI Travel Agent

This project is a conversational travel planner built with Python, LangGraph, Pydantic, and Rich.
It gathers a trip request, generates an itinerary, searches real flight and hotel offers through
Amadeus, ranks them against user preferences, and persists the profile, trip, and audit trail in
MongoDB.

## Production data and booking policy

- `TRAVEL_DATA_MODE=production` is the default. Missing Amadeus credentials or provider failures
  are reported explicitly; mock offers are never silently presented as real inventory.
- Set `TRAVEL_DATA_MODE=mock` only for local tests and demonstrations.
- Booking is an explicit hosted-checkout boundary. The agent must use a configured provider adapter
  and must not collect or store card numbers, CVV, expiry, or other raw payment data.
- The current checkout adapter intentionally raises `BookingUnavailableError` until a provider
  contract and hosted/tokenized checkout account are configured.

## Persistence and audit logging

Set `MONGODB_URI` and the provider secrets in `.env` or your deployment secret manager. The CLI
does not ask for credentials. If `TRAVEL_AGENT_MASTER_PASSWORD` is configured, the loaded values
are also encrypted at `~/.travel-agent/credentials.enc`; otherwise they remain managed by the
backend environment. The application creates indexes for
profiles, trips, and audit events. Audit events include correlation IDs and redacted excerpts; API keys,
passwords, tokens, email addresses, and payment fields are never retained in event data.

The local profile ID is stored at `~/.travel-agent/profile-id` unless `TRAVEL_PROFILE_ID` is set.
Credentials continue to use the encrypted store under `~/.travel-agent`.

## Providers

The initial provider boundaries are:

- Amadeus: flight and hotel search, with account-dependent booking capabilities
- Places provider: destination enrichment
- Weather provider: forecast data
- Currency provider: exchange rates

Provider adapters normalize external responses before they reach LangGraph nodes. Real offers must
be revalidated before booking because prices and availability can expire.

## Configuration

Copy `.env.example` to a secure environment configuration and provide (environment values override
the encrypted CLI store):

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=travel_agent
TRAVEL_DATA_MODE=production
AUDIT_RETENTION_DAYS=365
```

Run the CLI with:

```bash
.venv/bin/python main.py --provider openrouter
```

The test suite can run without MongoDB because repository and audit tests use `InMemoryStorage`.
