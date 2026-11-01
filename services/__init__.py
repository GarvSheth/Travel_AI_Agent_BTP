"""Package logs stay silent unless the app configures logging (NullHandler)."""
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
