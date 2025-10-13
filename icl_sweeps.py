import jax.numpy as jnp
from jax import random, grad
from typing import Any, Callable, Sequence
import jax
import optax
import matplotlib.pyplot as plt
import numpy as np
from jax.example_libraries import optimizers
import seaborn as sns
import sys
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description=''
    '''
    tests of multiple-layer-per-block models
    ''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--lr', default=5.0e-1, type=float, help='learning rate')
parser.add_argument('--frozen_embs',action='store_true')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--depth', type=int, default = -1)
parser.add_argument('--beta', type=float, default=1.0,
                         help='scaling factor for the residual branch. To use together with res_scaling parameter')
parser.add_argument('--gamma_zero', type=float, default=1.0,
                         help='controls the amount of feature learning.')
parser.add_argument('--lamb', type=float, default=1.0e-14,
                         help='regularization')
parser.add_argument('--steps',type=int, default = 100000)

parser.add_argument('--alpha_data',type=float, default = 1.5)
parser.add_argument('--beta_data',type=float, default = 1.75)
parser.add_argument('--data_dim',type=int, default = 64)
parser.add_argument('--data_P_tr',type=int, default = 128)
parser.add_argument('--data_P_te',type=int, default = 16)

args = parser.parse_args()

# set sweeps
if args.depth == -1:
    depths = [1,2,4,6,8,16,32,64]
else:
    depths = [args.depth]


if args.width == -1:
    widths = [4,8,16,32,64,128]
else:
    widths = [args.width]

sigma_in = 1.0
sigma_att = 1.0
sigma_out = 1.0
    
# data generating process
def sample_data(d, B, P_tr, P_te, seed = 0):

  X = random.normal(random.PRNGKey(2*seed), (B, P_tr + P_te , d))
  betas = random.normal( random.PRNGKey(2*seed + 1), (B,d) )
  y = jnp.einsum('ijk,ik->ij', X, betas) / jnp.sqrt(d) # B x (P + P_test)
  return X, y

def sample_data_spec_rotate(spec, w_star, B, P_tr, P_te, seed = 0):

  d = spec.shape[0]
  O = jnp.linalg.qr( random.normal(random.PRNGKey(2*seed), (B, d, d)) )[0]
  #O =  random.normal(random.PRNGKey(2*seed), (B, d, d)) / jnp.sqrt(d)
  X = random.normal(random.PRNGKey(2*seed+1), (B, P_tr + P_te , d))
  X = jnp.einsum('ijk,k->ijk', X, spec**(0.5))
  X = jnp.einsum('ijk,ilk->ijl', X , O)


  #betas = random.normal( random.PRNGKey(2*seed + 1), (B,d) )
  betas = jnp.einsum('ijk,k->ij', O, w_star)
  y = jnp.einsum('ijk,ik->ij', X, betas)  # B x (P + P_test)

  return X, y

sample_data_spec_rotate = jax.jit( sample_data_spec_rotate , static_argnums= (2,3,4))

# model initialization
"""
def init_params(d, N, seed = 0):

  W_x = sigma_in * random.normal(random.PRNGKey(seed), (N,d))
  W_y = sigma_in * random.normal(random.PRNGKey(seed+1), (N,))

  
  Wq = sigma_att*random.normal(random.PRNGKey(seed+2), (N,N))
  Wk = sigma_att*random.normal(random.PRNGKey(seed+3), (N,N))
  Wv = sigma_att*random.normal(random.PRNGKey(seed+4), (N,N))
  
  w_out = sigma_out * random.normal(random.PRNGKey(seed+5), (N,))

  params = [ W_x, W_y, Wq , Wk, Wv, w_out]
  return params
"""    

def init_params(d, N, sigma = 0.4):

  #W_x = jnp.sqrt(2.0) * sigma * random.normal(random.PRNGKey(0), (N,d))
  
  W_x = jnp.sqrt(2.0) * jnp.sqrt(N) * sigma * jnp.eye(N)
    
  W_y = jnp.ones(N) 

  Wq = sigma * jnp.eye(N) * jnp.sqrt(N)

  ## Wq = Wq - jnp.outer( W_y , W_y ) @ Wq / jnp.sum(W_y**2)

  #Wk = sigma *  random.normal(random.PRNGKey(3), (N,N))
  Wk = 1.0 * Wq
    
  #Wv = sigma *  random.normal(random.PRNGKey(4), (N,N))
  Wv = sigma * jnp.eye(N) * jnp.sqrt(N)
  #Wv = sigma * jnp.outer(W_y, W_y) / jnp.sum(W_y**2) 


  #w_out = sigma *  random.normal(random.PRNGKey(5), (N,))
  w_out = 1.0 * W_y

  params = [ W_x, W_y, Wq , Wk, Wv, w_out]
  return params
    

# X is B x seq x d
def model_eval(params, X, y, L=100, P_test = 1, beta = 100.0, qk_ln = False, norm_inputs = False):

  W_x, W_y, Wq, Wk, Wv, w_out = params

  N , d = W_x.shape

  # load in the input data
  h = jnp.einsum('ijk,lk->ijl', X, W_x)
  
  seq_len = X.shape[1]
  P_tr = seq_len - P_test

  # mask y data
  mask_y = np.ones( y.shape )
  mask_y[:,P_tr:] = np.zeros(( B, P_test ))

  h = h + jnp.einsum('ij,k->ijk', y * mask_y , W_y) # B x P x N


  mask = np.ones((seq_len, seq_len))
  mask[:,P_tr:] = np.zeros( (seq_len , P_test ))
  mask = jnp.array(mask)
 
  
  for l in range(L):
   
    q = jnp.einsum('ijk,lk->ijl', h,  Wq) / jnp.sqrt(N) # contract over N dimension
    k = jnp.einsum('ijk,lk->ijl', h,  Wk) / jnp.sqrt(N)

    if qk_ln:
      q = q - q.mean(axis = -1, keepdims = True)
      k = k - k.mean(axis = -1, keepdims = True)
      q = q / jnp.sqrt( jnp.mean( q**2 , axis = -1, keepdims = True ) )
      k = k / jnp.sqrt( jnp.mean( k**2 , axis = -1, keepdims = True ) )

    v = jnp.einsum('ijk,lk->ijl', h,  Wv) / jnp.sqrt(N)
    A = jnp.einsum('ijk,ilk->ijl', k, q) / N   # B x P x P


    # mask the attention
    h = h - beta/L * jnp.einsum('ijk,ilj->ilk', v,  jnp.einsum('ijk,jk->ijk', A, mask)  ) / P_tr

  #out = jnp.einsum('ijk,k->ij', h, w_out) / N
  out = jnp.einsum('ijk,k->ij', h, W_y) / N
  return out , [], []

def model_eval_decoupled(params, X, y, L=100, P_test = 1, beta = 100.0, qk_ln = False, norm_inputs = False):

  W_x, W_y, Wq, Wk, Wv, w_out = params

  N , d = W_x.shape

  # load in the input data
  hx = jnp.einsum('ijk,lk->ijl', X, W_x)
  
  seq_len = X.shape[1]
  P_tr = seq_len - P_test

  # mask y data
  mask_y = np.ones( y.shape )
  mask_y[:,P_tr:] = np.zeros(( B, P_test ))

  hy = jnp.einsum('ij,k->ijk', y * mask_y , W_y) # B x P x N


  mask = np.ones((seq_len, seq_len))
  mask[:,P_tr:] = np.zeros( (seq_len , P_test ))
  mask = jnp.array(mask)

  
  for l in range(L):
    
    q = jnp.einsum('ijk,lk->ijl', hx,  Wq) / jnp.sqrt(N) # contract over N dimension
    k = jnp.einsum('ijk,lk->ijl', hx,  Wk) / jnp.sqrt(N)

    v = jnp.einsum('ijk,lk->ijl', hy,  Wv) / jnp.sqrt(N)
    A = jnp.einsum('ijk,ilk->ijl', k, q) / N   # B x P x P


    # mask the attention
    hy = hy - beta/L * jnp.einsum('ijk,ilj->ilk', v,  jnp.einsum('ijk,jk->ijk', A, mask)  ) / P_tr

  #out = jnp.einsum('ijk,k->ij', hy, w_out) / N
  out = jnp.einsum('ijk,k->ij', hy, W_y) / N
  return out , [], []

# Uses Wx, Wq, Wk, Wv separate (i.e DECOUPLED)
# is not the most-reduced Gamma version
# Weights are fixed over layer iterations (RECURRENT)
def model_eval_decoupled_frozen_emb(params_tr, Wy , X, y, L=100, P_test = 1, beta = 100.0, qk_ln = False, norm_inputs = False):

  W_x, Wq, Wk, Wv = params_tr

  N , d = W_x.shape

  # load in the input data
  hx = jnp.einsum('ijk,lk->ijl', X, W_x)
  
  seq_len = X.shape[1]
  P_tr = seq_len - P_test

  # mask y data
  mask_y = np.ones( y.shape )
  mask_y[:,P_tr:] = np.zeros(( B, P_test ))

  hy = jnp.einsum('ij,k->ijk', y * mask_y , Wy) # B x P x N


  mask = np.ones((seq_len, seq_len))
  mask[:,P_tr:] = np.zeros( (seq_len , P_test ))
  mask = jnp.array(mask)

  
  for l in range(L):
    
    q = jnp.einsum('ijk,lk->ijl', hx,  Wq) / jnp.sqrt(N) # contract over N dimension
    k = jnp.einsum('ijk,lk->ijl', hx,  Wk) / jnp.sqrt(N)

    v = jnp.einsum('ijk,lk->ijl', hy,  Wv) / jnp.sqrt(N)
    A = jnp.einsum('ijk,ilk->ijl', k, q) / N   # B x P x P


    # mask the attention
    hy = hy - beta/L * jnp.einsum('ijk,ilj->ilk', v,  jnp.einsum('ijk,jk->ijk', A, mask)  ) / P_tr

  #out = jnp.einsum('ijk,k->ij', hy, w_out) / N
  out = jnp.einsum('ijk,k->ij', hy, Wy) / N
  return out , [], []

def train_model( data_params, model_params , opt_params, spec = None, w_star = None, save_path = None):

  d, P_tr, P_test, B = data_params

  N, L, beta, gamma = model_params

  T, lr, lamb = opt_params


  params = init_params(d, N)
  
  qk_ln = False
  #loss_fn = lambda pt, X, y: jnp.mean( ( model_eval(pt, X, y, L = L, P_test=P_test, beta = beta, qk_ln=False)[0][:,P_tr:] / gamma + y[:,P_tr:] )**2 )

  loss_fn = lambda pt, X, y: jnp.mean( ( model_eval_decoupled(pt, X, y, L = L, P_test=P_test, beta = beta, qk_ln=False)[0][:,P_tr:] / gamma + y[:,P_tr:] )**2 )
    
  W_x, Wy, Wq, Wk, Wv, w_out = params
  if args.frozen_embs:
    loss_fn = lambda pt, X, y: jnp.mean( ( model_eval_decoupled_frozen_emb(pt, Wy, X, y, L = L, P_test=P_test, beta = beta, qk_ln=False)[0][:,P_tr:] / gamma + y[:,P_tr:] )**2 )
     

  loss_fn = jax.jit(loss_fn)

  reg_loss_fn = lambda pt, X, y: N * gamma**2 * loss_fn(pt, X, y) + lamb * optimizers.l2_norm(pt)**2



  #schedule = lambda t: lr/(1.0 + t)**(0.1)
  opt_init, opt_update, get_params = optimizers.sgd( lr )
  
  
  if args.frozen_embs:
    params_tr = [W_x, Wq, Wk, Wv]
    opt_state = opt_init( params_tr )
  else:
    opt_state = opt_init(params)


  loss_grad_fn = jax.jit( jax.value_and_grad(reg_loss_fn) )


  jit_sample = jax.jit(lambda seed: sample_data_spec_rotate(spec, w_star, B, P_tr, P_test, seed = seed))
  pretrain_loss = []
  cos_A_Wy = []
  cos_Wy_wout = []
  cos_Wv_Wy = []
  norms_Wy = []
  norms_Wv = []
  norms_Wx = []
  norms_Wk = []
  for t in range(T):
    if spec is not None:
      #X , y = sample_data_spec_rotate(spec, w_star, B, P_tr, P_test, seed = t)
      X, y = jit_sample(t)
    else:
      X, y = sample_data(d, B, P_tr, P_test, seed = t)
    _ , grads = loss_grad_fn(get_params(opt_state), X, y)
    loss = loss_fn(get_params(opt_state), X, y)
    opt_state = opt_update(t, grads, opt_state)
    if t % 100 == 0:
        sys.stdout.write(f'\r Loss step {t}: {loss}')
        
        #wandb.log({'loss': loss})
    
    if args.frozen_embs:
        Wx, Wq, Wk, Wv = get_params(opt_state) 
    else:
        Wx, Wy, Wq, Wk, Wv, w_out = get_params(opt_state)

        
    
    A_mat = Wk.T @ Wq / N
    cos_A_Wy += [ jnp.dot( Wy, A_mat @ Wy ) / jnp.sqrt( jnp.sum( A_mat**2 ) ) / jnp.sum(Wy**2) ]
    cos_Wy_wout +=  [ jnp.dot(Wy, w_out) / jnp.sqrt( jnp.sum(Wy**2) * jnp.sum(w_out**2) )  ]     
    cos_Wv_Wy += [  jnp.dot( Wy, Wv @ Wy ) / jnp.sqrt( jnp.sum(Wv**2) ) /jnp.sum(Wy**2)  ] 
    norms_Wy += [ jnp.sqrt( jnp.dot(Wy,Wy) )  ]
    norms_Wv += [ jnp.sqrt( jnp.sum(Wv**2) ) ]
    norms_Wx += [ jnp.sqrt( jnp.sum(Wx**2) ) ]
    norms_Wk += [ jnp.sqrt( jnp.sum(Wk**2) ) ] 
    if t % 1000 == 0 and save_path != None:
        np.save(save_path, pretrain_loss)
        all_aligns = np.array(  [ cos_A_Wy, cos_Wy_wout, cos_Wv_Wy, norms_Wy, norms_Wv, norms_Wx, norms_Wk ] )
        np.save( save_path +'_aligns', all_aligns )
                
    pretrain_loss += [ loss ]

  return pretrain_loss


def get_run_name(args):
    return "width_{}/depth_{}/frozen_embs_{}/lr_{:.4f}/steps_{}/batch_size_{}/beta_{}/gamma_zero_{}/lamb_{}/alpha_data_{}/beta_data_{}/data_dim_{}/data_P_tr_{}".format(
    args.width,args.depth,args.frozen_embs,args.lr,args.steps,args.batch_size,args.beta,args.gamma_zero,args.lamb,args.alpha_data,args.beta_data,args.data_dim, args.data_P_tr)

save_dir = '/n/netscratch/pehlevan_lab/Lab/bbordelon/ICL_sweeps/'



d = args.data_dim
P_tr = args.data_P_tr
P_test = args.data_P_te
B = args.batch_size
#N = #

T = args.steps
lr =  args.lr

lamb = args.lamb

alpha = args.alpha_data
spec = jnp.linspace(1,d,d)**(- alpha)

beta = args.beta_data
w_star = jnp.sqrt( jnp.linspace(1, d, d)**(-alpha*beta-1.0) / spec )
w_star = w_star / jnp.sqrt( jnp.sum( w_star**2 * spec ) )
beta_model = args.beta

data_params = [d, P_tr, P_test, B]
opt_params = [T, lr, lamb]

all_losses_L = []
gamma = args.gamma_zero


for i, N in enumerate(widths):
    for j, L in enumerate(depths):
        
        model_params = [ N, L, beta_model, gamma ]
        
        args.width = N
        args.depth = L
        run_name = get_run_name(args)

        """
        wandb.init(
                    # set the wandb project where this run will be logged
                    project="ICL_project",
                    # track hyperparameters and run metadata
                    config=args.__dict__)
        wandb.run.name = run_name
        """  
        save_path = os.path.join(save_dir, run_name.replace("/", "-"))

        losses = train_model(data_params, model_params, opt_params, spec = spec, w_star = w_star, save_path=save_path)
        
        #np.save(save_path, losses)
                
        
