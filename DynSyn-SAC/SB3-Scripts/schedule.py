from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return a linear learning rate schedule with optional warmup.

    Args:
        initial_value: Initial learning rate.

    Returns:
        A function that takes progress_remaining (1 -> 0) and returns
        the current learning rate.
    """
    def func(progress_remaining: float) -> float:
        """Compute learning rate from remaining progress (1 at start, 0 at end)."""
        warmup_fraction = 0.00
        warmup_end = 1 - warmup_fraction
        min_lr_ratio = 0.25
        if progress_remaining > warmup_end:
            # Warmup phase
            warmup_progress = (1 - progress_remaining) / warmup_fraction
            return warmup_progress * warmup_end * initial_value
        else:
            if progress_remaining < min_lr_ratio:
                return min_lr_ratio * initial_value
            else:
                # Linear decay after warmup
                return progress_remaining * initial_value

    return func