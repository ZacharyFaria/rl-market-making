"""Probabilistic Level-2 Limit Order Queue Reconstructor with Back-of-Queue Bias.

Implements REQ-SIM-002: Estimates limit order queue position from Level-2 Market-By-Price
(MBP) volume updates using a cancellation probability decay function parameterized by
a constant $k \\in (-1, 0]$.
"""

from __future__ import annotations


def estimate_queue_position(
    volume_ahead: float,
    k: float,
    cancelled_volume: float | None = None,
    total_level_volume: float | None = None,
) -> float:
    """Calculate reconstructed limit order queue position under back-of-queue bias $k$.

    In Level-2 Market-By-Price (MBP) feeds, individual order cancellation IDs are not
    observable. Empirical Market-By-Order (L3) studies show that cancellations cluster
    near the back of the queue. Parameterizing the cancellation probability decay by
    $k \\in (-1, 0]$ assigns a higher cancellation likelihood to the back of the queue
    as $k$ becomes more negative, thereby reducing the fraction of cancellations that
    deplete volume *ahead* of our order:

    .. math::
        p_{\\text{front}}(x; k) = x^{1 - k}, \\qquad -1 < k \\le 0

    where $x \\in (0, 1]$ is the relative queue position. Consequently, a more negative
    $k$ leaves a strictly larger effective `volume_ahead` (slower queue advancement)
    after a Level-2 volume reduction.

    Parameters
    ----------
    volume_ahead : float
        Current aggregate resting volume ahead of the agent's order at its price tier.
    k : float
        Back-of-queue cancellation bias parameter. Must strictly satisfy $-1 < k \\le 0$.
        Values outside $(-1, 0]$ raise a `ValueError` (`REQ-SIM-002`).
    cancelled_volume : float | None, optional
        Inferred volume reduction at the price tier. If `None`, defaults to a standard
        $20\\%$ fractional tick cancellation of `volume_ahead`.
    total_level_volume : float | None, optional
        Total resting volume at the price tier prior to the cancellation update.
        If `None`, defaults to `1.5 * max(volume_ahead, 1.0)`.

    Returns
    -------
    float
        Updated effective volume ahead of the agent's order (`>= 0.0`).
    """
    if k <= -1.0 or k > 0.0:
        raise ValueError(
            f"Queue bias parameter k={k} is outside the valid mathematical domain (-1, 0] (REQ-SIM-002)."
        )
    if volume_ahead < 0.0:
        raise ValueError(f"volume_ahead must be non-negative, got {volume_ahead}.")
    if volume_ahead == 0.0:
        return 0.0

    total_vol = (
        float(total_level_volume)
        if total_level_volume is not None and total_level_volume > volume_ahead
        else 1.5 * float(volume_ahead)
    )
    cancel_vol = (
        float(cancelled_volume)
        if cancelled_volume is not None
        else 0.25 * float(volume_ahead)
    )
    cancel_vol = max(0.0, min(cancel_vol, total_vol))

    # Relative position in queue x in (0, 1]; exponent (1 - k) in [1, 2)
    rel_pos = min(max(float(volume_ahead) / max(total_vol, 1e-8), 1e-6), 1.0)
    front_cancel_prob = rel_pos ** (1.0 - float(k))

    depleted_ahead = cancel_vol * front_cancel_prob
    updated_ahead = max(0.0, float(volume_ahead) - depleted_ahead)
    return updated_ahead

