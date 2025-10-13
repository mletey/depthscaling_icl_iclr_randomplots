# R1 setting
# Reduced Gamma model (no explicit ws)
# power law data
# Unrolled

import jax.numpy as jnp
# from jax import random, grad, lax
# from typing import Any, Callable, Sequence
import jax
# import optax
from jax import jit

# from flax import linen as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from jax.example_libraries import optimizers

from jax.tree_util import tree_map
from jax import value_and_grad

import sys

print("imports done", flush=True)

def reduced_gamma_unrolled(spec, beta_bar, N, L, B, K, P, sigma = 0.0, eta = 0.01, T=100, lamb = 1e-3, rotate=True, ctx_sample=True):

    # ----- dims -----
    d = len(spec)

    # ----- fixed A (shared across layers, not trained) -----
    A = jnp.zeros((N, d))
    m = min(N, d)
    A = A.at[:m, :m].set(jnp.eye(m) * jnp.sqrt(N))  # same as your original

    # ----- trainable per-layer Gammas -----
    def init_gammas():
        return [jnp.zeros((N, N)) for _ in range(L)]

    Gammas = init_gammas()  # list of length L

    # ----- loss: only depends on Gammas (A is fixed) -----
    @jit
    def compute_loss(Gammas, A, Os_t, Sigmas_t, betas_t):
        batch, dim, _ = Sigmas_t.shape
        vs = 1.0 * betas_t
        I = jnp.eye(dim)

        for l in range(L):
            AGA = (1.0 / N) * (A.T @ Gammas[l] @ A)            # (d x d)
            M_l = jnp.broadcast_to(I, (batch, dim, dim)) - (1.0 / L) * jnp.einsum("jk,ikl->ijl", AGA, Sigmas_t)
            vs = jnp.einsum("ijk,ik->ij", M_l, vs)

        Ov = jnp.einsum("ijk,ij->ik", Os_t, vs)
        reg = sum(jnp.mean(G**2) for G in Gammas)             # L2 on Gammas only
        return jnp.einsum("ik,ik,k->", Ov, Ov, spec) / batch + lamb * reg

    loss_and_grad = jit(value_and_grad(compute_loss, argnums=0))  # grads wrt Gammas only

    losses = []
    Gammas = init_gammas()

    for t in range(T):
        # --- your sampling code (O, Sigma_c, betas) stays as-is ---
        O = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3*t+1), (B, d, d)))[0]
        if ctx_sample:
            X = jax.random.normal(jax.random.PRNGKey(3*t), (B, P, d))
            X = jnp.einsum('ijk,k->ijk', X, spec**0.5)
            OX = jnp.einsum('ijk,ilk->ilj', O, X)
            Sigma_c = jnp.einsum('ijk,ijl->ikl', OX, OX) / P
        else:
            Sigma_c = jnp.einsum('ijk,ilk,k->ijl', O, O, spec)

        bernoulli = jax.random.bernoulli(jax.random.PRNGKey(3*t+2), shape=(B, d))
        bernoulli = 2.0 * bernoulli - 1.0
        w_sign = w_star[jnp.newaxis, :] * bernoulli
        betas = jnp.einsum('ijk,ik->ij', O, w_sign)
        # ------------------------------------------------------------

        loss_t, grads = loss_and_grad(Gammas, A, O, Sigma_c, betas)  # grads is a list like Gammas
        Gammas = tree_map(lambda G, g: G - eta * g, Gammas, grads)   # SGD step on Γ_l only
        losses.append(float(loss_t))

    return losses


# myname = sys.argv[1] # grab value of $mydir to add results
mydir = sys.argv[1]
Ls = [1, 2, 4, 8, 16]
layerindex = int(sys.argv[2])-1
L = Ls[layerindex]

M = 45
N = 256
K = 16
B = 512
P = 512

T = int(sys.argv[3])

alpha = 1.75
beta = 0.5

print("parameters done", flush=True)

spec = jnp.linspace(1,M,M)**(-alpha)
w_star = jnp.linspace(1,M,M)**(- (alpha*beta+1 - alpha)*0.5 )
w_star= w_star / jnp.sqrt( jnp.sum(w_star**2 * spec)  )

print("calling train", flush=True)

losses = reduced_gamma_unrolled(spec, w_star, N, L, B, K, P, T = T, eta = 3.0)

with open(f"{mydir}/output_{layerindex}.txt", "w") as f:
    f.write(f"{losses}")
