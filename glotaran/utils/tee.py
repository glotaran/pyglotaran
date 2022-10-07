"""Module containing context manager to capture stdout output and still forward it to stdout."""
from __future__ import annotations

import sys
from io import StringIO
from types import TracebackType


class TeeContext:
    """Context manager that allows to work with string written to stdout.

    This context manager behaves similar to the ``tee`` shell command.
    https://linuxize.com/post/linux-tee-command
    """

    def __init__(self) -> None:
        """Create new ``StringIO`` buffer and save reference to original ``sys.stdout``."""
        self.buffer = StringIO()
        self.stdout = sys.stdout

    def __enter__(self) -> TeeContext:
        """Replace ``sys.stdout`` on entering the context.

        Returns
        -------
        TeeContext
            Instance that can be read from.
        """
        sys.stdout = self  # type:ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Restore ``sys.stdout`` on exiting the context."""
        sys.stdout = self.stdout
        return None

    def write(self, data: str) -> None:
        """Write to buffer and original ``sys.stdout``.

        Parameters
        ----------
        data: str
            String to write to stdout and buffer.
        """
        self.buffer.write(data)
        self.stdout.write(data)

    def read(self) -> str:
        """Return values written to buffer.

        Returns
        -------
        str
            Text written to buffer.
        """
        return self.buffer.getvalue()

    def flush(self) -> None:
        """Flush values in the buffer."""
        self.buffer.flush()
