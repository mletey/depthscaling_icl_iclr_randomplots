# SOFTMAX ATTENTION, UNROLLED setting
# power law data

import numpy as np
import sys
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn import softmax, gelu
import optax

print("imports done", flush=True)


def init_params_tr_layers(d, N, L, sigma=0.4, key=None):
    """
    Initialize parameters for L layers.

    Each layer l has its own (W_x^l, Wq^l, Wk^l, Wv^l).
    Returns:
        params_tr_layers: list of length L with tuples (W_x, Wq, Wk, Wv)
        Wy: readout vector (N,)
    """
    # Base template
    W_x_base = jnp.sqrt(2.0) * jnp.sqrt(N) * sigma * jnp.eye(N, d)  # shape (N,d)
    Wq_base  = sigma * jnp.eye(N) * jnp.sqrt(N)
    Wk_base  = 1.0 * Wq_base
    Wv_base  = sigma * jnp.eye(N) * jnp.sqrt(N)
    Wy       = jnp.ones(N)

    params_tr_layers = [(W_x_base, Wq_base, Wk_base, Wv_base) for _ in range(L)]

    return params_tr_layers, Wy
    

def init_params_tr_layers_MLP(d, N, L, sigma=0.4, key=None):
    """
    Initialize parameters for L layers.

    Each layer l has its own (W_x^l, Wq^l, Wk^l, Wv^l, W_mlp1^l, W_mlp2^l).
    Returns:
        params_tr_layers: list of length L with tuples (W_x, Wq, Wk, Wv, W_mlp1, W_mlp2)
        Wy: readout vector (N,)
    """
    # Base templates
    W_x_base   = jnp.sqrt(2.0) * jnp.sqrt(N) * sigma * jnp.eye(N, d)  # (N,d)
    Wq_base    = sigma * jnp.eye(N) * jnp.sqrt(N)
    Wk_base    = 1.0   * Wq_base
    Wv_base    = sigma * jnp.eye(N) * jnp.sqrt(N)
    W_mlp1_base= sigma * jnp.eye(N)                                   # (N,N)
    W_mlp2_base= sigma * jnp.eye(N)                                   # (N,N)
    Wy         = jnp.ones(N)

    params_tr_layers = []
    if key is not None:
        keys = jr.split(key, L * 6)
        for l in range(L):
            W_x    = W_x_base    + 0.01 * jr.normal(keys[6*l+0], (N, d))
            Wq     = Wq_base     + 0.01 * jr.normal(keys[6*l+1], (N, N))
            Wk     = Wk_base     + 0.01 * jr.normal(keys[6*l+2], (N, N))
            Wv     = Wv_base     + 0.01 * jr.normal(keys[6*l+3], (N, N))
            W_mlp1 = W_mlp1_base + 0.01 * jr.normal(keys[6*l+4], (N, N))
            W_mlp2 = W_mlp2_base + 0.01 * jr.normal(keys[6*l+5], (N, N))
            params_tr_layers.append((W_x, Wq, Wk, Wv, W_mlp1, W_mlp2))
    else:
        params_tr_layers = [
            (W_x_base, Wq_base, Wk_base, Wv_base, W_mlp1_base, W_mlp2_base)
            for _ in range(L)
        ]

    return params_tr_layers, Wy

# -------------------------------
# Unrolled model with softmax attn
# -------------------------------
def model_eval_decoupled_unrolled(
    params_tr_layers,  # list of length L: [(W_x^l, Wq^l, Wk^l, Wv^l), ...]
    Wy,                # (N,)
    X,                 # (B, P, d)
    y,                 # (B, P)
    P_test=1,
    beta=100.0,
    qk_ln=False,
    norm_inputs=False
):
    L = len(params_tr_layers)
    W_x0, _, _, _ = params_tr_layers[0]
    N, d = W_x0.shape

    B = y.shape[0]
    P = X.shape[1]
    P_tr = P - P_test

    # masks
    mask_y = jnp.ones_like(y)
    mask_y = mask_y.at[:, P_tr:].set(0.0)  # zero test labels

    # label embeddings
    hy = jnp.einsum('ij,k->ijk', y * mask_y, Wy)  # (B,P,N)

    # token mask (P,P); 1=keep, 0=mask
    mask = jnp.ones((P, P))
    mask = mask.at[:, P_tr:].set(0.0)

    neg_inf = jnp.array(-1e9, dtype=hy.dtype)

    for l in range(L):
        W_x, Wq, Wk, Wv = params_tr_layers[l]

        # project inputs once per layer from raw X
        hx = jnp.einsum('ijk,lk->ijl', X, W_x)  # (B,P,N)

        q = jnp.einsum('ijk,lk->ijl', hx, Wq) / jnp.sqrt(N)
        k = jnp.einsum('ijk,lk->ijl', hx, Wk) / jnp.sqrt(N)
        v = jnp.einsum('ijk,lk->ijl', hy, Wv) / jnp.sqrt(N)

        # logits and masked softmax over "keys" axis=1
        A_logits = jnp.einsum('ijk,ilk->ijl', k, q) / N  # (B,P,P) indexed (i,j,l)
        A_logits = A_logits + (1.0 - mask)[None, :, :] * neg_inf
        A_attn   = softmax(A_logits, axis=1)             # (B,P,P)

        # update hy
        hy = hy - (beta / L) * jnp.einsum('ijk,ijl->ilk', v, A_attn) #/ (P - P_test)

    out = jnp.einsum('ijk,k->ij', hy, Wy) / N  # (B,P)
    return out, [], []

def model_eval_decoupled_unrolled_WITHMLP(
    params_tr_layers,   # [(W_x, Wq, Wk, Wv, W_mlp1, W_mlp2), ...]
    Wy,                 # (N,)
    X,                  # (B,P,d)
    y,                  # (B,P)
    P_test=1,
    beta=100.0,
    qk_ln=False,
    norm_inputs=False
):
    L = len(params_tr_layers)
    W_x0, *_ = params_tr_layers[0]
    N, d = W_x0.shape

    B = y.shape[0]
    P = X.shape[1]
    P_tr = P - P_test

    mask_y = jnp.ones_like(y)
    mask_y = mask_y.at[:, P_tr:].set(0.0)
    hy = jnp.einsum('ij,k->ijk', y * mask_y, Wy)  # (B,P,N)

    mask = jnp.ones((P, P))
    mask = mask.at[:, P_tr:].set(0.0)
    neg_inf = jnp.array(-1e9, dtype=hy.dtype)

    for l in range(L):
        W_x, Wq, Wk, Wv, W_mlp1, W_mlp2 = params_tr_layers[l]

        # Input projection
        hx = jnp.einsum('ijk,lk->ijl', X, W_x)  # (B,P,N)

        # Attention
        q = jnp.einsum('ijk,lk->ijl', hx, Wq) / jnp.sqrt(N)
        k = jnp.einsum('ijk,lk->ijl', hx, Wk) / jnp.sqrt(N)
        v = jnp.einsum('ijk,lk->ijl', hy, Wv) / jnp.sqrt(N)

        A_logits = jnp.einsum('ijk,ilk->ijl', k, q) / N   # (B,P,P) index (i,j,l)
        A_logits = A_logits + (1.0 - mask)[None, :, :] * neg_inf
        A_attn   = softmax(A_logits, axis=1)

        # Attention update (keeps your original /P_tr scaling)
        hy = hy - (beta / L) * jnp.einsum('ijk,ijl->ilk', v, A_attn) 

        # 2-layer MLP with GELU on channel dim (square weights N×N)
        h_hidden = gelu(jnp.einsum('ijk,lk->ijl', hy, W_mlp1))/jnp.sqrt(N)       # (B,P,N)
        hy       = hy - (beta / L) * jnp.einsum('ijk,lk->ijl', h_hidden, W_mlp2)/jnp.sqrt(N)

    out = jnp.einsum('ijk,k->ij', hy, Wy) / N
    return out, [], []
# ---------------------------------
# Training function (Optax AdamW)
# ---------------------------------

def train_model(
    X, y,
    data_params,            # (d, P_tr, P_test, B)
    model_params,           # (N, L, beta, gamma)
    opt_params,             # (T, lr, lamb)
):
    d, P_tr, P_test, B = data_params
    N, L, beta, gamma = model_params
    T, lr, weight_decay = opt_params

    nonwy, Wy = init_params_tr_layers(d, N, L, sigma=0.4, key=None)

    # Pack params
    params = {
        "layers": nonwy,  # list[(W_x, Wq, Wk, Wv)]
        "Wy": Wy
    }

    # ---------- loss ----------
    def loss_fn(params, X, y):
        out, _, _ = model_eval_decoupled_unrolled(
            params["layers"], params["Wy"], X, y,
            P_test=P_test, beta=beta, qk_ln=False
        )
        pred_test   = out[:, P_tr:] / gamma
        target_test = y[:, P_tr:]
        return jnp.mean((pred_test + target_test) ** 2)

    # ---------- optimizer with per-parameter LR ----------
    # Two transforms: "base" (lr), "attn" (lr * L)
    transforms = {
        "base": optax.adamw(learning_rate=lr,     weight_decay=weight_decay),
        "attn": optax.adamw(learning_rate=lr, weight_decay=weight_decay),
        # "mlp": optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    }

    # Build a labels pytree that mirrors params' structure:
    # - For each layer tuple (W_x, Wq, Wk, Wv) => ("base","attn","attn","attn")
    # - Wy => "base"
    labels = {
        #"layers": [("base", "attn", "attn", "attn", "mlp", "mlp")] * len(params["layers"]),
        "layers": [("base", "attn", "attn", "attn")] * len(params["layers"]),
        "Wy": "base",
    }

    tx = optax.multi_transform(transforms, labels)
    opt_state = tx.init(params)

    @jit
    def train_step(params, opt_state, X, y):
        loss_val, grads = value_and_grad(loss_fn)(params, X, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    loss_history = []
    for _ in range(T):
        params, opt_state, loss_val = train_step(params, opt_state, X, y)
        loss_history.append(float(loss_val))

    return params, loss_history

def sample_data_spec_rotate(spec, w_star, B, P_tr, P_te, seed = 0):
  d = spec.shape[0]
  O = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3*seed), (B, d, d)) )[0]
  X = jax.random.normal(jax.random.PRNGKey(3*seed+1), (B, P_tr+P_te, d))
  X = jnp.einsum('ijk,k->ijk', X, spec**(0.5))
  OX = jnp.einsum('ijk,ilk->ilj', O, X)
  bernoulli = jax.random.bernoulli(jax.random.PRNGKey(3*seed+2), shape=(B,d))
  bernoulli = 2*bernoulli - 1.0
  w_sign = w_star[jnp.newaxis,:] * bernoulli
  y = jnp.einsum('ijk,ik->ij', X, w_sign)
  return OX, y

def draw_pretraining_data(B, P_tr, P_te, rho, C):
    d = C.shape[0]
    x = np.random.randn(B, P_tr+P_te, d) / np.sqrt(d)
    w_set = np.random.multivariate_normal(mean=np.zeros(d), cov=C, size = B)
    epsilon = np.random.randn(B,  P_tr+P_te) * np.sqrt(rho)
    y = np.einsum('nij,nj->ni', x, w_set) + epsilon
    return x, y 

# d = int(sys.argv[1])
# P_tr = int(sys.argv[2])
# P_test = int(sys.argv[3])
# B = int(sys.argv[4])
# T = int(sys.argv[5])
# lr = float(sys.argv[6])
# lamb = float(sys.argv[7])
# alpha = float(sys.argv[8])
# spec = jnp.linspace(1,d,d)**(- alpha)
# beta = float(sys.argv[9])
# w_star = jnp.sqrt( jnp.linspace(1, d, d)**(-alpha*beta-1.0) / spec )
# w_star = w_star / jnp.sqrt( jnp.sum( w_star**2 * spec ) )
# beta_model = float(sys.argv[10])
# data_params = [d, P_tr, P_test, B]
# opt_params = [T, lr, lamb]
# gamma = float(sys.argv[11])
# L = int(sys.argv[12])
# N = int(sys.argv[13])
# model_params = [ N, L, beta_model, gamma ]

mydir = sys.argv[1]

B = 512
N = 128

Ls = [1, 2, 4, 8, 16]
layerindex = int(sys.argv[2])-1
L = Ls[layerindex]

beta_model = 1.0
gamma = 1.0
lamb = 0.000000000000001
T = int(sys.argv[3])
lr = float(sys.argv[4])
d = 64
P_tr = 128
P_test = 16
alpha = 1.5
beta = 1.75
spec = jnp.linspace(1,d,d)**(- alpha)
w_star = jnp.sqrt( jnp.linspace(1, d, d)**(-alpha*beta-1.0) / spec )
w_star = w_star / jnp.sqrt( jnp.sum( w_star**2 * spec ) )
data_params = [d, P_tr, P_test, B]
opt_params = [T, lr, lamb]
model_params = [ N, L, beta_model, gamma ]

print("parameters done", flush=True)

X, y = draw_pretraining_data(B, P_tr, P_test, 0.01, np.eye(d)) #sample_data_spec_rotate(spec, w_star, B, P_tr, P_test, seed = 64)
print("sampling done", flush=True)

_, losses = train_model(X,y,data_params, model_params, opt_params)
print("training done", flush=True)

with open(f"{mydir}/output_{layerindex}.txt", "w") as f:
    f.write(f"{losses}")

