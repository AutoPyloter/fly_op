"""Fly-Proposer variant (PROTOCOL.md 1.2 #1, Faz 1): produces a new
candidate solution from the substrate's spiking response to the current x.

Encoding: x is injected as constant current into `dim` neurons drawn from
`encode_pool` (an array of eligible neuron indices). Decoding: mean firing
rate of a `n_readout`-neuron subset drawn from `decode_pool`, linear
readout to Δx.

`encode_pool`/`decode_pool` default to None, meaning "the whole
population" — this was Faz 1's design (PROTOCOL.md encoding/decoding
option (a)/(b) relaxed, classification data not available yet; see
preregistration/faz_1.md). Faz 1b passes the real afferent/efferent
neuron index arrays (preregistration/faz_1b.md) to test the protocol's
actual encoding (a) ("gerçek duyusal nörona enjeksiyon") and decoding (a)
("motor nöron popülasyonundan readout") alternatives.

Encode/decode neuron indices and the readout matrix are drawn from `seed`
only, never from the substrate, so the same seed gives every substrate
(real connectome + every null) an identical proposal mechanism — the
substrate's dynamics is the only thing that can differ (PROTOCOL.md 4.1).

`readout_lr > 0` (Faz 1c, preregistration/faz_1c.md) turns on a cheap,
restricted form of plasticity: a reward-modulated three-factor Hebbian
update on the readout matrix ONLY (pre-synaptic factor = firing rate,
post-synaptic factor = the delta it produced, third factor = whether the
resulting candidate improved on the current solution). The connectome's
own synapses are still never touched — this isolates "can a *trained
interpreter* extract something useful from the fixed dynamics" from full
Fly-RL (PROTOCOL.md 2.5), which is far more expensive and lets plasticity
reach the substrate itself. `readout_lr=0.0` (default) reproduces Faz
1/1b exactly — no behavior change unless explicitly opted into.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyopt.substrate import SearchContext, Substrate


@dataclass(frozen=True)
class FlyProposerConfig:
    dim: int
    n_readout: int = 50
    T: int = 15
    step_scale: float = 1.0
    decode_scale: float = 0.5
    readout_lr: float = 0.0  # 0.0 = frozen readout (Faz 1/1b); >0 = Faz 1c plasticity


class FlyProposer:
    def __init__(
        self,
        substrate: Substrate,
        config: FlyProposerConfig,
        seed: int,
        encode_pool: np.ndarray | None = None,
        decode_pool: np.ndarray | None = None,
    ):
        self.substrate = substrate
        self.config = config
        rng = np.random.default_rng(seed)

        if encode_pool is None and decode_pool is None:
            # Faz 1 default: shared pool, sampled without overlap.
            chosen = rng.choice(substrate.n_units, size=config.dim + config.n_readout, replace=False)
            self.encode_indices = chosen[: config.dim]
            self.decode_indices = chosen[config.dim :]
        else:
            # Faz 1b: distinct real pools (e.g. afferent / efferent) —
            # disjoint by biology, not by construction, but assert it to
            # catch a bad pool file rather than silently double-counting a
            # neuron as both sensory input and motor output.
            enc_pool = encode_pool if encode_pool is not None else np.arange(substrate.n_units)
            dec_pool = decode_pool if decode_pool is not None else np.arange(substrate.n_units)
            assert not set(enc_pool.tolist()) & set(dec_pool.tolist()), "encode_pool and decode_pool overlap"
            self.encode_indices = rng.choice(enc_pool, size=config.dim, replace=False)
            self.decode_indices = rng.choice(dec_pool, size=config.n_readout, replace=False)

        self.readout_W = rng.normal(
            0, 1.0 / np.sqrt(config.n_readout), size=(config.dim, config.n_readout)
        )
        self._last_firing_rate: np.ndarray | None = None
        self._last_delta: np.ndarray | None = None

    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray:
        state = self.substrate.reset(seed=ctx.iteration)
        stim = np.zeros(self.substrate.n_units, dtype=np.float32)
        stim[self.encode_indices] = x.astype(np.float32) * self.config.step_scale
        spike_accum = np.zeros(self.substrate.n_units)
        for _ in range(self.config.T):
            spikes, state = self.substrate.step(stim, state)
            spike_accum += spikes
        firing_rate = spike_accum[self.decode_indices] / self.config.T
        delta = self.readout_W @ firing_rate
        if self.config.readout_lr > 0:
            self._last_firing_rate = firing_rate
            self._last_delta = delta
        return x + self.config.decode_scale * delta

    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None:
        if self.config.readout_lr <= 0 or self._last_firing_rate is None:
            return  # no plasticity (Faz 1/1b default, unlike Fly-RL PROTOCOL.md 2.5)
        # reward-modulated three-factor Hebbian update on the readout only:
        # pre=firing_rate, post=delta produced, third factor=reward. Decay
        # keeps ||readout_W|| bounded instead of growing unboundedly over
        # the search's iterations.
        reward = 1.0 if improved else -1.0
        lr = self.config.readout_lr
        self.readout_W *= 1.0 - lr
        self.readout_W += lr * reward * np.outer(self._last_delta, self._last_firing_rate)
