"""Normalize the formatting around string tool arguments."""


def normalize_tool_input(value: str) -> str:
    """Remove outer whitespace and one matching pair of wrapper quotes.

    Leave internal whitespace, punctuation, and escape sequences untouched:
    arguments may contain SMILES, JSON, or multiline code.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1].strip()
    return value


def normalize_molecule_input(value: str) -> str:
    value = normalize_tool_input(value)
    if not value:
        raise ValueError("Empty molecule input; provide one molecule name or SMILES.")
    if value.startswith("{"):
        raise ValueError(
            "Invalid molecule input: provide one plain molecule name or SMILES, "
            "not a JSON object or query wrapper."
        )
    return value
