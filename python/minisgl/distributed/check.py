from __future__ import annotations


def check_matching_settings(settings_by_rank: list[dict[str, object]]) -> None:
    """Report fields that would make TP workers execute incompatible operations."""
    reference = settings_by_rank[0]
    for rank, settings in enumerate(settings_by_rank[1:], start=1):
        different = sorted(
            name
            for name in reference.keys() | settings.keys()
            if name not in reference or name not in settings or reference[name] != settings[name]
        )
        if different:
            raise ValueError(f"TP rank {rank} has mismatched settings: {', '.join(different)}")
