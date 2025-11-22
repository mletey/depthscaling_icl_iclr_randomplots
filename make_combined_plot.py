import matplotlib.pyplot as plt
import jax.numpy as jnp
import seaborn as sns
import ast

sns.set(font_scale=2)
sns.set_style("whitegrid")
sns.set_palette("rocket", n_colors=10)
colors = sns.color_palette("rocket", n_colors=10)
Lvals = [1, 2, 4, 8, 16]
# alpha = 1.75
# beta = 0.5

experiments = ['SOFTMAX_T20000', 'SOFTMAXMLP_T20000_lr0p0001', 'MHA_T20000_lr0p0001_isotropic', 'MHAMLP_T20000_lr0p0001_isotropic']
parents = ['outputs', 'outputs', 'rebuttals', 'rebuttals']
titles = ['Softmax only', 'Softmax and MLP', 'MHA Softmax only', 'MHA Softmax and MLP']
fig, axs = plt.subplots(1, 4, figsize=(48, 10))

for runind in range(len(experiments)):
    all_loss = []
    for i in range(len(Lvals)):
        with open(f"{parents[runind]}/{experiments[runind]}/output_{i}.txt", "r") as f:
            data = f.read()
            all_loss.append(ast.literal_eval(data))
    for i, loss in enumerate(all_loss):
        axs[runind].loglog(loss, label = f'$L = {Lvals[i]}$', color=colors[i+3])
    axs[runind].set_xlabel(r'$t$',fontsize = 20)
    axs[runind].set_title(titles[runind])

handles, labels = axs[0].get_legend_handles_labels()
axs[0].set_ylabel(r'$\mathcal{L}(t)$',fontsize = 20)
# Add a single legend above all subplots
fig.legend(handles, labels,
           loc='upper center',        # place legend at top center
           bbox_to_anchor=(0.5, 1),# adjust vertical position
           ncol=len(labels),          # put entries in a single row
           frameon=False)             # optional: no box

plt.savefig(f"figures/combined.pdf", bbox_inches = 'tight')
