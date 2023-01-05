class GlotaranDefinitionError(Exception):
    pass


class GlotaranModelError(Exception):
    pass


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


class ModelItemIssue(ItemIssue):
    """Issue for missing model items."""

    def __init__(self, item_name: str, label: str):
        """Create a model issue.

        Parameters
        ----------
        item_name : str
            The name of the item.
        label : str
            The item label.
        """
        self._item_name = item_name
        self._label = label

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return f"Missing model item '{self._item_name}' with label '{self._label}'."


class ParameterIssue(ItemIssue):
    """Issue for missing parameters."""

    def __init__(self, label: str):
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
