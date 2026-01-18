from __future__ import annotations


class GlotaranDefinitionError(Exception):
    pass


class GlotaranModelError(Exception):
    pass


class GlotaranUserError(Exception):
    pass


class GlotaranModelIssues(GlotaranModelError):  # noqa: N818
    def __init__(self, issues: list[ItemIssue]) -> None:
        joiner = "\n* "
        super().__init__(
            f"The model has issues:\n\n* {joiner.join([i.to_string() for i in issues])}"
        )


class ItemIssue:
    """Baseclass for item issues."""

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError

    def __rep__(self) -> str:
        """Get the representation."""
        return self.to_string()


class ParameterIssue(ItemIssue):
    """Issue for missing parameters."""

    def __init__(self, label: str) -> None:
        """Create a parameter issue.

        Parameters
        ----------
        label : str
            The parameter label.
        """
        self._label = label

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return f"Missing parameter with label '{self._label}'."
