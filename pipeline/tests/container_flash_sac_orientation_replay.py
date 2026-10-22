"""Prove the thirteenth raw term reaches n-step replay and reward relabeling.

The plan's first review focus: a reward patch that only changed a scalar would
leave FlashSAC's learning targets untouched, because the learner recomposes
one-step and n-step rewards from *stored raw objective terms* and the current
curriculum weights. These tests drive NVIDIA's own replay code with thirteen
terms, so they fail if any part of that path is sized to twelve.
Container-only (JAX).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from flash_chord.training.flash_sac.replay import (
    ReplayBatch,
    ReplayInsert,
    ReplaySpec,
    create_replay,
    flush_pending,
    gather_replay,
    insert_replay,
    relabel_rewards,
    replay_memory,
)

from pipeline.flash_sac_orientation import ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES

ALL_TERMS = len(ORIENTATION_RECON_BODY_OBJECTIVE_TERM_NAMES)
FRAME_DT = 0.02
DISCOUNT = 0.99
N_STEP = 7
WORLDS = 8


def spec(term_count: int = ALL_TERMS, n_step: int = N_STEP) -> ReplaySpec:
    return ReplaySpec(
        capacity=4096,
        minimum_size=1,
        batch_size=16,
        n_step=n_step,
        discount=DISCOUNT,
        world_count=WORLDS,
        observation_dim=486,
        action_dim=43,
        objective_term_count=term_count,
        critic_context_dim=10,
        observation_storage_dtype="float16",
    )


def make_insert(terms: np.ndarray, done: bool = False) -> ReplayInsert:
    observation = jnp.zeros((WORLDS, 486), dtype=jnp.float32)
    context = jnp.zeros((WORLDS, 10), dtype=jnp.float32)
    flag = jnp.full((WORLDS,), done, dtype=jnp.bool_)
    return ReplayInsert(
        observation=observation,
        action=jnp.zeros((WORLDS, 43), dtype=jnp.float32),
        objective_terms=jnp.asarray(terms, dtype=jnp.float32),
        terminated=flag,
        truncated=jnp.zeros((WORLDS,), dtype=jnp.bool_),
        next_observation=observation,
        critic_context=context,
        next_critic_context=context,
    )


# --- the buffers really are thirteen wide -----------------------------------


def test_replay_allocates_thirteen_term_storage() -> None:
    state = create_replay(spec())
    assert state.storage.discounted_objective_terms.shape[-1] == ALL_TERMS
    assert state.pending.objective_terms.shape[-1] == ALL_TERMS


def test_extra_term_costs_exactly_four_bytes_per_transition() -> None:
    """The plan's memory claim: 4 B/transition, 1.91 MiB in a 500k ring."""
    delta = replay_memory(spec(13)).bytes_per_transition - replay_memory(spec(12)).bytes_per_transition
    assert delta == 4
    ring_delta = replay_memory(spec(13)).ring_bytes - replay_memory(spec(12)).ring_bytes
    assert ring_delta == 4 * 4096


# --- relabeling uses all thirteen weights -----------------------------------


def batch_of(terms) -> ReplayBatch:
    """A real ``ReplayBatch`` -- ``relabel_rewards`` is jitted and needs a pytree."""
    rows = terms.shape[0]
    observation = jnp.zeros((rows, 486), dtype=jnp.float32)
    context = jnp.zeros((rows, 10), dtype=jnp.float32)
    flag = jnp.zeros((rows,), dtype=jnp.bool_)
    return ReplayBatch(
        observation=observation,
        action=jnp.zeros((rows, 43), dtype=jnp.float32),
        discounted_objective_terms=terms,
        terminated=flag,
        truncated=flag,
        next_observation=observation,
        critic_context=context,
        next_critic_context=context,
    )


def test_relabel_rewards_consumes_the_thirteenth_term() -> None:
    rng = np.random.default_rng(3)
    terms = jnp.asarray(rng.normal(size=(32, ALL_TERMS)), dtype=jnp.float32)
    weights = jnp.asarray(rng.normal(size=ALL_TERMS), dtype=jnp.float32)
    reward = relabel_rewards(batch_of(terms), weights, jnp.float32(FRAME_DT))
    expected = FRAME_DT * np.asarray(terms) @ np.asarray(weights)
    assert np.max(np.abs(np.asarray(reward) - expected)) < 1e-4


def test_changing_only_the_thirteenth_weight_changes_the_learning_target() -> None:
    """The decisive check that this is not a scalar-only reward patch."""
    rng = np.random.default_rng(5)
    terms = jnp.asarray(rng.normal(size=(64, ALL_TERMS)), dtype=jnp.float32)
    base = jnp.zeros(ALL_TERMS, dtype=jnp.float32)
    off = np.asarray(relabel_rewards(batch_of(terms), base, jnp.float32(FRAME_DT)))
    on = np.asarray(relabel_rewards(batch_of(terms), base.at[-1].set(2.0), jnp.float32(FRAME_DT)))
    difference = on - off
    assert np.max(np.abs(difference)) > 1e-6
    assert np.allclose(difference, FRAME_DT * 2.0 * np.asarray(terms)[:, -1], atol=1e-5)


def test_relabel_rejects_a_stale_twelve_weight_vector() -> None:
    terms = jnp.zeros((4, ALL_TERMS), dtype=jnp.float32)
    with pytest.raises(ValueError):
        relabel_rewards(batch_of(terms), jnp.zeros(12, dtype=jnp.float32), jnp.float32(FRAME_DT))


# --- end-to-end through NVIDIA's own n-step accumulation --------------------


def test_n_step_window_discounts_the_thirteenth_term_like_the_others() -> None:
    """Push a full n-step window and read back the stored discounted terms."""
    rng = np.random.default_rng(9)
    window = rng.normal(size=(N_STEP, WORLDS, ALL_TERMS))
    state = create_replay(spec())
    for step in range(N_STEP):
        state = insert_replay(state, make_insert(window[step]))
    state = flush_pending(state)

    # The oldest transition has the complete n-step window behind it.
    stored = np.asarray(gather_replay(state, jnp.arange(WORLDS)).discounted_objective_terms)
    expected = sum(DISCOUNT**k * window[k] for k in range(N_STEP))
    assert stored.shape == (WORLDS, ALL_TERMS)
    assert np.max(np.abs(stored - expected)) < 1e-4
    # Specifically the appended slot, which is the whole point.
    assert np.max(np.abs(stored[:, -1] - expected[:, -1])) < 1e-4


def test_a_terminal_transition_truncates_the_thirteenth_term_too() -> None:
    """``n_step=2`` so one window commits deterministically on the second insert."""
    rng = np.random.default_rng(11)
    window = rng.normal(size=(2, WORLDS, ALL_TERMS))
    state = create_replay(spec(n_step=2))
    state = insert_replay(state, make_insert(window[0], done=True))
    state = insert_replay(state, make_insert(window[1]))
    state = flush_pending(state)
    stored = np.asarray(gather_replay(state, jnp.arange(WORLDS)).discounted_objective_terms)
    # Done at the first step: the window stops there, no bootstrap into step 1.
    assert np.max(np.abs(stored - window[0])) < 1e-4


def test_stored_terms_are_raw_not_weighted() -> None:
    """Replay must hold the unweighted, un-frame-scaled value."""
    terms = np.full((WORLDS, ALL_TERMS), 0.0)
    terms[:, -1] = -0.25  # a 90 degree error
    state = create_replay(spec(n_step=1))
    state = insert_replay(state, make_insert(terms))
    state = flush_pending(state)
    stored = np.asarray(gather_replay(state, jnp.arange(WORLDS)).discounted_objective_terms)
    assert np.allclose(stored[:, -1], -0.25, atol=1e-6)
    # Only after relabeling does the weight and frame_dt appear.
    weights = jnp.zeros(ALL_TERMS, dtype=jnp.float32).at[-1].set(2.0)
    reward = np.asarray(relabel_rewards(gather_replay(state, jnp.arange(WORLDS)), weights, jnp.float32(FRAME_DT)))
    assert np.allclose(reward, -0.01, atol=1e-6)
