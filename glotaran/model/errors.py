from __future__ import annotations


class GlotaranDefinitionError(Exception):
    pass


class GlotaranModelError(Exception):
    """Raised for a model that is inconsistent, for example elements with different dimensions."""


class GlotaranUserError(Exception):
    """Raised for input that cannot be used, for example a dataset label without data."""


class GlotaranModelIssues(GlotaranModelError):  # noqa: N818
    """Raised with the list of the problems of a model that are found when a fit starts."""

    def __init__(self, issues: list[ItemIssue]) -> None:
        """Create the error from the issues of a model.

        Parameters
        ----------
        issues : list[ItemIssue]
            The issues, which are listed in the message of the error.
        """
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
