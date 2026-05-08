"""LandscapeTracker — lightweight energy-history logger used during folding."""


class LandscapeTracker:
    """Records per-call energy values and named stage boundaries."""

    def __init__(self) -> None:
        self.history: list[float] = []
        self.stage_markers: list[tuple[int, str]] = []
        self.current_iter: int = 0

    def log(self, energy: float) -> None:
        self.history.append(energy)
        self.current_iter += 1

    def mark_stage(self, name: str) -> None:
        self.stage_markers.append((self.current_iter, name))
