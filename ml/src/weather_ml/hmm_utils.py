"""Lightweight Gaussian HMM utilities (diagonal covariance)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class HmmParams:
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    covars: np.ndarray


def _gaussian_pdf(obs: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    diff = obs[:, None, :] - means[None, :, :]
    var = covars[None, :, :]
    log_prob = -0.5 * (
        np.sum(diff**2 / var, axis=2) + np.sum(np.log(2 * np.pi * var), axis=2)
    )
    log_prob = log_prob - np.max(log_prob, axis=1, keepdims=True)
    return np.exp(log_prob)


def _forward_backward(
    obs: np.ndarray, params: HmmParams
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_steps, _ = obs.shape
    n_states = params.A.shape[0]
    B = _gaussian_pdf(obs, params.means, params.covars)
    B = np.clip(B, 1e-12, None)

    alpha = np.zeros((n_steps, n_states), dtype=float)
    beta = np.zeros_like(alpha)
    scale = np.zeros(n_steps, dtype=float)

    alpha[0] = params.pi * B[0]
    scale[0] = 1.0 / np.sum(alpha[0])
    alpha[0] *= scale[0]
    for t in range(1, n_steps):
        alpha[t] = (alpha[t - 1] @ params.A) * B[t]
        scale[t] = 1.0 / np.sum(alpha[t])
        alpha[t] *= scale[t]

    beta[-1] = 1.0 * scale[-1]
    for t in range(n_steps - 2, -1, -1):
        beta[t] = (params.A @ (B[t + 1] * beta[t + 1])) * scale[t]

    gamma = alpha * beta
    xi = np.zeros((n_steps - 1, n_states, n_states), dtype=float)
    for t in range(n_steps - 1):
        denom = np.sum(alpha[t][:, None] * params.A * B[t + 1] * beta[t + 1])
        if denom <= 0:
            continue
        xi[t] = (
            alpha[t][:, None] * params.A * B[t + 1] * beta[t + 1] / denom
        )
    log_likelihood = -np.sum(np.log(scale + 1e-12))
    return gamma, xi, float(log_likelihood)


def fit_gaussian_hmm(
    obs: np.ndarray,
    *,
    n_states: int = 2,
    n_iters: int = 10,
    seed: int = 0,
) -> HmmParams:
    if obs.ndim != 2:
        raise ValueError("Observations must be 2D array.")
    n_steps, n_dim = obs.shape
    if n_steps < n_states:
        raise ValueError("Not enough observations to fit HMM.")

    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(obs)
    means = np.zeros((n_states, n_dim), dtype=float)
    covars = np.zeros_like(means)
    pi = np.zeros(n_states, dtype=float)
    for state in range(n_states):
        mask = labels == state
        if not mask.any():
            means[state] = obs.mean(axis=0)
            covars[state] = obs.var(axis=0) + 1e-3
            pi[state] = 1.0 / n_states
        else:
            means[state] = obs[mask].mean(axis=0)
            covars[state] = obs[mask].var(axis=0) + 1e-3
            pi[state] = float(mask.mean())

    A = np.full((n_states, n_states), 1.0 / n_states, dtype=float)
    A += np.eye(n_states) * 0.2
    A = A / A.sum(axis=1, keepdims=True)

    params = HmmParams(pi=pi, A=A, means=means, covars=covars)

    for _ in range(n_iters):
        gamma, xi, _ = _forward_backward(obs, params)
        gamma_sum = gamma.sum(axis=0)
        pi = gamma[0]
        A = xi.sum(axis=0)
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        means = (gamma.T @ obs) / np.maximum(gamma_sum[:, None], 1e-12)
        diff = obs[:, None, :] - means[None, :, :]
        covars = (
            (gamma[:, :, None] * diff**2).sum(axis=0)
            / np.maximum(gamma_sum[:, None], 1e-12)
        )
        covars = np.maximum(covars, 1e-3)
        params = HmmParams(pi=pi, A=A, means=means, covars=covars)

    return params


def forward_filter(obs: np.ndarray, params: HmmParams) -> np.ndarray:
    n_steps, _ = obs.shape
    n_states = params.A.shape[0]
    B = _gaussian_pdf(obs, params.means, params.covars)
    B = np.clip(B, 1e-12, None)
    alpha = np.zeros((n_steps, n_states), dtype=float)
    alpha[0] = params.pi * B[0]
    alpha[0] /= np.sum(alpha[0])
    for t in range(1, n_steps):
        alpha[t] = (alpha[t - 1] @ params.A) * B[t]
        total = np.sum(alpha[t])
        if total > 0:
            alpha[t] /= total
    return alpha
