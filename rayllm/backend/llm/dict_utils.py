def merge_dicts(overwrite: dict, base: dict) -> dict:
    """
    Merge two dictionaries recursively, with keys from overwrite taking precedence.
    """
    base = base.copy()
    for key, value in overwrite.items():
        if isinstance(value, dict):
            # get node or create one
            node = base.setdefault(key, {})
            merge_dicts(value, node)
        else:
            base[key] = value

    return base
