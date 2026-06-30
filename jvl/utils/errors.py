class JVLError(Exception):
    """Base exception JVLIVS."""


class BackendNotAvailable(JVLError):
    """Backend LLM non joignable."""


class ConfigError(JVLError):
    """Erreur de configuration."""


class ModelNotFound(JVLError):
    """Modèle introuvable sur le backend."""
