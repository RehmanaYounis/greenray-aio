"""Exceptions shared by the shell and the command emulators."""


class CommandError(Exception):
    """Raised by a command to print ``message`` and exit with a non-zero ``code``."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code
