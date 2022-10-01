"""Glotaran module with regular expression patterns and functions."""
import re


class RegexPattern:
    """An 'Enum' of (compiled) regular expression patterns (rp)."""

    # tuple = re.compile(r"(\(.*?,.*?\))")
    elements_in_string_of_list: re.Pattern = re.compile(r"(\(.+?\)|[-+.\d]+)")
    group: re.Pattern = re.compile(r"(\(.+?\))")
    list_with_tuples: re.Pattern = re.compile(r"(\[.+\(.+\).+\])")
    word: re.Pattern = re.compile(r"[\w]+")
    number_scientific: re.Pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)")
    number: re.Pattern = re.compile(r"[\d.+-]+")
    tuple_word: re.Pattern = re.compile(r"(\([.\s\w\d]+?[,.\s\w\d]*?\))")
    tuple_number: re.Pattern = re.compile(r"(\([\s\d.+-]+?[,\s\d.+-]*?\))")
    optimization_stdout: re.Pattern = re.compile(
        r"^\s+(?P<iteration>\d+)\s+(?P<nfev>\d+)"
        r"\s+(?P<cost>\d\.\d+e[+-]\d+)"
        r"(\s+(?P<cost_reduction>\d\.\d+e[+-]\d+)\s+(?P<step_norm>\d\.\d+e[+-]\d+)|\s+)"
        r"\s+(?P<optimality>\d\.\d+e[+-]\d+)\s*?$",
        re.MULTILINE,
    )
