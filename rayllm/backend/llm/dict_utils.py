def merge_dicts(base: dict, overwrite: dict) -> dict:
    """
    Merge overwrite into base. Modify base inplace.
    """

    for key in overwrite:
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(overwrite[key], dict)
        ):
            merge_dicts(base[key], overwrite[key])
        else:
            base[key] = overwrite[key]
    return base
