import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Load the three JSON files
files = {
    'Standard': '/home/daniel/l/masteruppsats/np-reg/image/results/stability_metrics_vanilla.json',
    'BatchNorm': '/home/daniel/l/masteruppsats/np-reg/image/results/stability_metrics_bn.json',
    'NP': '/home/daniel/l/masteruppsats/np-reg/image/results/stability_metrics_np.json',
}

data = {}
for label, filepath in files.items():
    with open(filepath) as f:
        data[label] = json.load(f)

def to_min_max(series):
    if not series:
        return np.array([]), np.array([])
    if isinstance(series[0], dict):
        mins = np.array([item.get('min', np.nan) for item in series])
        maxs = np.array([item.get('max', np.nan) for item in series])
    else:
        mins = np.array([min(values) for values in series])
        maxs = np.array([max(values) for values in series])
    return mins, maxs

# Extract metrics
metrics = {}
for label, data_dict in data.items():
    loss_landscapes = data_dict.get('loss_landscapes', [])
    gradient_stabilities = data_dict.get('gradient_stabilities', [])
    effective_beta_smoothness = data_dict.get('effective_betas', [])

    loss_mins, loss_maxs = to_min_max(loss_landscapes)

    # Gradient predictiveness uses the min-max range across probe steps.
    gradient_mins, gradient_maxs = to_min_max(gradient_stabilities)

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
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'medium',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.titlesize': 13,
    'axes.titleweight': 'medium',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
})
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

colors = {'Standard': '#E07070', 'BatchNorm': '#5B8BC5', 'NP': '#70B070'}
effective_beta_stride = 10
loss_stride = 10
gradient_stride = 10
x_tick_step = 2000
legend_fontsize = 11
legend_handlelength = 2.0
legend_order = ['Standard', 'BatchNorm', 'NP']


def apply_legend_order(ax, order, **legend_kwargs):
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    ordered_handles = [label_to_handle[label] for label in order if label in label_to_handle]
    ordered_labels = [label for label in order if label in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, **legend_kwargs)

# (a) Loss landscape variation
ax = axes[0]
for label in ['NP', 'Standard', 'BatchNorm']:
    steps = np.arange(len(metrics[label]['loss_maxs']))[::loss_stride]
    loss_mins = metrics[label]['loss_mins'][::loss_stride]
    loss_maxs = metrics[label]['loss_maxs'][::loss_stride]
    ax.fill_between(steps, loss_mins, loss_maxs, color=colors[label], label=label, alpha=0.7)

ax.set_xlabel('Steps')
ax.set_ylabel('Loss Landscapes')
max_steps_a = max(len(metrics[label]['loss_maxs']) for label in metrics)
ax.set_xticks(np.arange(0, max_steps_a + 1, x_tick_step))
ax.set_xlim(0, max_steps_a - 1)
ax.set_title('(a) Loss Landscapes')
apply_legend_order(
    ax,
    legend_order,
    loc='upper right',
    fontsize=legend_fontsize,
    handlelength=legend_handlelength,
)
ax.set_yscale('log')
ax.set_ylim(0.6,4)
ax.set_yticks([1, 3])
ax.set_yticklabels([r'$10^0$', r'$3\times10^0$'])
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

# (b) Gradient predictiveness
ax = axes[1]
for label in ['Standard','NP',  'BatchNorm']:
    steps = np.arange(len(metrics[label]['gradient_maxs']))[::gradient_stride]
    gradient_mins = metrics[label]['gradient_mins'][::gradient_stride]
    gradient_maxs = metrics[label]['gradient_maxs'][::gradient_stride]
    ax.fill_between(steps, gradient_mins, gradient_maxs, color=colors[label], label=label, alpha=0.6)

ax.set_xlabel('Steps')
ax.set_ylabel('Gradient Predictiveness')
ax.set_title('(b) Gradient Predictiveness')
max_steps_b = max(len(metrics[label]['gradient_maxs']) for label in metrics)
ax.set_xticks(np.arange(0, max_steps_b + 1, x_tick_step))
ax.set_xlim(0, max_steps_b - 1)
ax.set_ylim(0,2.8)
ax.set_yticks([0, 1, 2])
apply_legend_order(
    ax,
    legend_order,
    loc='upper right',
    fontsize=legend_fontsize,
    handlelength=legend_handlelength,
)

# (c) Effective beta smoothness
ax = axes[2]
for label in ['NP', 'Standard','BatchNorm']:
    data_m_full = metrics[label]['effective_beta_smoothness']
    data_m = data_m_full[::effective_beta_stride]
    steps = np.arange(len(data_m_full))[::effective_beta_stride]
    sns.lineplot(x=steps, y=data_m, ax=ax, color=colors[label], linewidth=1, label=label, alpha=0.9)

ax.set_xlabel('Steps')
ax.set_ylabel(r'effective $\beta$-smoothness')
ax.set_title(r'(c) Effective $\beta$-smoothness')
max_steps_c = max(len(metrics[label]['effective_beta_smoothness']) for label in metrics)
ax.set_xticks(np.arange(0, max_steps_c + 1, x_tick_step))
ax.set_xlim(0, max_steps_c - 1)
apply_legend_order(
    ax,
    legend_order,
    loc='upper right',
    fontsize=legend_fontsize,
    handlelength=legend_handlelength,
)
ax.set_ylim(0,2.8)
ax.set_yticks([0, 1, 2])

plt.tight_layout()
plt.savefig('/home/daniel/l/masteruppsats/np-reg/image/results/stability_metrics_plots.pdf', bbox_inches='tight', pad_inches=0.02)
print("Plot saved to stability_metrics_plots.png")
plt.close()
