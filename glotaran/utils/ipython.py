"""Glotaran module with utilities for ``ipython`` integration (e.g. ``notebooks``)."""

from collections import UserString


class MarkdownStr(UserString):
    """String wrapper class for rich display integration of markdown in ipython."""

    def __init__(self, wrapped_str: str, *, syntax: str = None):
        """String class automatically displayed as markdown by ipython.


        Parameters
        ----------
        wrapped_str: str
            String to be wrapped.
        syntax: str
            Syntax highlighting which should be applied, by default None

        Note
        ----
        Possible syntax highlighting values can e.g. be found here:
        https://support.codebasehq.com/articles/tips-tricks/syntax-highlighting-in-markdown
        """
        # This needs to be called data since ipython is looking for this attr
        self.data = str(wrapped_str)
        self.syntax = syntax

    def _repr_markdown_(self) -> str:
        """Special method used by ``ipython`` to render markdown.

        See:
        https://ipython.readthedocs.io/en/latest/config/integrating.html?highlight=_repr_markdown_#rich-display

        Returns
        -------
        str:
            Markdown string wrapped in a code block with syntax
            highlighting if syntax is not None.
        """
        if self.syntax is not None:
            return f"```{self.syntax}\n{self.data}\n```"
        else:
            return self.data

    def __str__(self) -> str:
        """Representation used by print and str."""
        return self._repr_markdown_()
