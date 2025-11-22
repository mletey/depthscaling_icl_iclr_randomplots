import matplotlib.pyplot as plt
import jax.numpy as jnp
import seaborn as sns
import ast

sns.set(font_scale=1.3)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)

experiment = input("experiment: ")
parent = input("parent: ")
T = int(input("T: "))
theoryline = int(input("theory line? "))

Lvals = [1, 2, 4, 8, 16]
alpha = 1.75
beta = 0.5

all_loss = []
for i in range(len(Lvals)):
  # with open(f"outputs/{experiment}/output_{i}.txt", "r") as f:
  #   data = f.read().strip()
  # values = data.strip("[]").split()
  # values = [float(v) for v in values]
  # all_loss.append(values)
  with open(f"{parent}/{experiment}/output_{i}.txt", "r") as f:
    data = f.read()
    all_loss.append(ast.literal_eval(data))

sns.set_palette("rocket", n_colors=len(Lvals))
for i, loss in enumerate(all_loss):
  plt.loglog(loss, label = f'$L = {Lvals[i]}$')
if theoryline != 0:  
    plt.loglog(jnp.linspace(10,T,T),  0.6* jnp.linspace(10,T,T)**(- beta/(2+beta) ) , '--',label = r'$t^{-\beta/(2+\beta)}$',color = 'blue')
plt.xlabel(r'$t$',fontsize = 20)
plt.ylabel(r'$\mathcal{L}(t)$',fontsize = 20)
plt.legend()
plt.savefig(f"figures/{experiment}_loss_vs_t.pdf", bbox_inches = 'tight')

plt.clf()

for i, loss in enumerate(all_loss):
  plt.loglog(Lvals[i] * jnp.linspace(1,len(loss),len(loss)), loss, label = f'$L = {Lvals[i]}$')
beta_t = beta/(2+beta)
if theoryline != 0:  
    plt.loglog(jnp.linspace(1,Lvals[i]*T,T), 0.5 * jnp.linspace(1,Lvals[i]*T,T)**(- beta/(beta+3.0) ) , '--', label = r'$C^{- \beta/(3+\beta) }$' , color = 'red')
plt.xlabel(r'$C$',fontsize = 20)
plt.ylabel(r'$\mathcal{L}(C)$',fontsize = 20)
plt.legend()
plt.savefig(f"figures/{experiment}_loss_vs_C.pdf", bbox_inches = 'tight')
plt.show()