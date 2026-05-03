import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the three JSON files
files = {
    'Standard': '/home/daniel/l/masteruppsats/np-reg/image/results/reg_cifar_stability_metrics_vanilla.json',
    'BatchNorm': '/home/daniel/l/masteruppsats/np-reg/image/results/reg_cifar_stability_metrics_bn.json',
    'NP': '/home/daniel/l/masteruppsats/np-reg/image/results/reg_cifar_stability_metrics_np.json',
}

data = {}
for label, filepath in files.items():
    with open(filepath) as f:
        data[label] = json.load(f)

# Extract metrics
metrics = {}
for label, data_dict in data.items():
    loss_landscapes = data_dict.get('loss_landscapes', [])
    gradient_stabilities = data_dict.get('gradient_relative_stabilities', [])
    effective_beta_smoothness = data_dict.get('effective_betas', [])
    
    loss_maxs = np.array([item['max'] for item in loss_landscapes])
    loss_mins = np.array([item['min'] for item in loss_landscapes])

    # Gradient predictiveness should be the min-max range.
    gradient_maxs = np.array([item['max'] for item in gradient_stabilities])
    gradient_mins = np.array([item['min'] for item in gradient_stabilities])

    # Effective beta-smoothness should come from the third top-level array.
    metrics[label] = {
        'loss_maxs': loss_maxs,
        'loss_mins': loss_mins,
        'gradient_maxs': gradient_maxs,
        'gradient_mins': gradient_mins,
        'effective_beta_smoothness': effective_beta_smoothness,
    }

# Create figure similar to Figure 4
sns.set_theme(style='whitegrid', context='paper')
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

colors = {'Standard': '#E07070', 'BatchNorm': '#5B8BC5', 'NP': '#70B070'}

# (a) Loss landscape variation
ax = axes[0]
for label in ['Standard', 'BatchNorm', 'NP']:
    steps = np.arange(len(metrics[label]['loss_maxs']))
    ax.fill_between(steps, metrics[label]['loss_mins'], metrics[label]['loss_maxs'], color=colors[label], label=label)
    sns.lineplot(x=steps, y=metrics[label]['loss_maxs'], ax=ax, color=colors[label], linewidth=1.5, legend=False)

ax.set_xlabel('Steps')
ax.set_ylabel('Loss Landscapes')
ax.set_title('(a) loss landscape')
ax.legend(loc='upper right')
ax.set_yscale('log')

# (b) Gradient predictiveness
ax = axes[1]
for label in ['Standard', 'BatchNorm', 'NP']:
    steps = np.arange(len(metrics[label]['gradient_maxs']))
    ax.fill_between(steps, metrics[label]['gradient_mins'], metrics[label]['gradient_maxs'], color=colors[label], label=label)
    sns.lineplot(x=steps, y=metrics[label]['gradient_maxs'], ax=ax, color=colors[label], linewidth=1.5, legend=False)

ax.set_xlabel('Steps')
ax.set_ylabel('Gradient Predictiveness')
ax.set_title('(b) gradient predictiveness')
ax.legend(loc='upper right')

# (c) Effective beta smoothness
ax = axes[2]
for label in ['Standard', 'BatchNorm', 'NP']:
    data_m = metrics[label]['effective_beta_smoothness']
    steps = np.arange(len(data_m))
    sns.lineplot(x=steps, y=data_m, ax=ax, color=colors[label], linewidth=1.5, label=label)

ax.set_xlabel('Steps')
ax.set_ylabel(r'$\beta$-smoothness')
ax.set_title(r'(c) effective $\beta$-smoothness')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/daniel/l/masteruppsats/np-reg/image/stability_metrics_plots.png', dpi=150, bbox_inches='tight')
print("Plot saved to stability_metrics_plots.png")
plt.close()
