import base64
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


PROVIDER_KEYS = {
    "gemini":      ["GEMINI_API_KEY",      "AMADEUS_API_KEY", "AMADEUS_API_SECRET", "MONGODB_URI"],
    "anthropic":   ["ANTHROPIC_API_KEY",   "AMADEUS_API_KEY", "AMADEUS_API_SECRET", "MONGODB_URI"],
    "openrouter":  ["OPENROUTER_API_KEY",  "AMADEUS_API_KEY", "AMADEUS_API_SECRET", "MONGODB_URI"],
}


class CredentialManager:
    CONFIG_DIR = Path.home() / ".travel-agent"
    CREDS_FILE = CONFIG_DIR / "credentials.enc"
    SALT_FILE = CONFIG_DIR / ".salt"
    PBKDF2_ITERATIONS = 480_000

    def __init__(self, provider: str = "gemini"):
        self.CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._credentials: dict[str, str] = {}
        self._is_unlocked = False
        self._fernet_key: bytes | None = None
        self.provider = provider

    @property
    def REQUIRED_KEYS(self) -> list[str]:
        return PROVIDER_KEYS.get(self.provider, PROVIDER_KEYS["gemini"])

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.PBKDF2_ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

    def _get_or_create_salt(self) -> bytes:
        if self.SALT_FILE.exists():
            return self.SALT_FILE.read_bytes()
        salt = os.urandom(16)
        self.SALT_FILE.write_bytes(salt)
        self.SALT_FILE.chmod(0o600)
        return salt

    def is_configured(self) -> bool:
        return self.CREDS_FILE.exists() and self.CREDS_FILE.stat().st_size > 0

    def load_from_environment(self) -> list[str]:
        """Load required backend secrets without interactive prompts.

        Returns missing variable names. Values remain in memory unless an optional
        master password is supplied by the deployment environment.
        """
        missing: list[str] = []
        values: dict[str, str] = {}
        for key in self.REQUIRED_KEYS:
            value = os.getenv(key, "").strip()
            if not value or value.startswith("...") or value.endswith("..."):
                missing.append(key)
            else:
                values[key] = value
        if missing:
            return missing
        self._credentials = values
        self._is_unlocked = True
        master_password = os.getenv("TRAVEL_AGENT_MASTER_PASSWORD", "").strip()
        if master_password:
            self._persist(values, master_password)
        return []

    def _persist(self, creds: dict[str, str], master_password: str) -> None:
        salt = self._get_or_create_salt()
        fernet_key = self._derive_key(master_password, salt)
        encrypted = Fernet(fernet_key).encrypt(json.dumps(creds).encode("utf-8"))
        self.CREDS_FILE.write_bytes(encrypted)
        self.CREDS_FILE.chmod(0o600)
        self._fernet_key = fernet_key

    def setup(self, master_password: str) -> None:
        """Persist values supplied by the backend environment."""
        missing = self.load_from_environment()
        if missing:
            raise ValueError(
                "Missing backend credentials: " + ", ".join(missing)
            )
        creds = dict(self._credentials)

        self._persist(creds, master_password)
        self._credentials = creds
        self._is_unlocked = True

    def unlock(self, master_password: str) -> None:
        """Decrypt credential store into memory. Raises ValueError on wrong password."""
        if not self.is_configured():
            raise RuntimeError("Credentials not configured. Run first-time setup.")
        salt = self._get_or_create_salt()
        fernet_key = self._derive_key(master_password, salt)
        f = Fernet(fernet_key)
        try:
            decrypted = f.decrypt(self.CREDS_FILE.read_bytes())
        except InvalidToken:
            raise ValueError("Incorrect master password.")
        self._credentials = json.loads(decrypted.decode("utf-8"))
        self._is_unlocked = True
        self._fernet_key = fernet_key

    def get_credential(self, name: str) -> str:
        if not self._is_unlocked:
            raise RuntimeError("Credential store is locked. Call unlock() first.")
        if name not in self._credentials:
            raise KeyError(f"Credential '{name}' not found.")
        return self._credentials[name]

    def has_credential(self, name: str) -> bool:
        return self._is_unlocked and bool(self._credentials.get(name))

    def set_unlocked_credential(self, name: str, value: str) -> None:
        """Add or update a value using the key from the current unlocked session."""
        if not self._is_unlocked or self._fernet_key is None:
            raise RuntimeError("Credential store is locked. Call unlock() first.")
        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")
        self._credentials[name] = value.strip()
        encrypted = Fernet(self._fernet_key).encrypt(
            json.dumps(self._credentials).encode("utf-8")
        )
        self.CREDS_FILE.write_bytes(encrypted)
        self.CREDS_FILE.chmod(0o600)

    def set_credential(self, name: str, value: str, master_password: str) -> None:
        """Update a single credential and re-encrypt the store."""
        if not self._is_unlocked:
            raise RuntimeError("Credential store is locked. Call unlock() first.")
        self._credentials[name] = value
        salt = self._get_or_create_salt()
        fernet_key = self._derive_key(master_password, salt)
        f = Fernet(fernet_key)
        encrypted = f.encrypt(json.dumps(self._credentials).encode("utf-8"))
        self.CREDS_FILE.write_bytes(encrypted)
