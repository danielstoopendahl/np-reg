import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
sns.set_theme(style='ticks', context='paper')
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

colors = {'Vanilla': '#C37238', 'Batch Norm': '#829750', 'NP-reg': '#789EB8'}
legend_order = ['Vanilla', 'Batch Norm', 'NP-reg']
legend_handlelength = 2.0

effective_beta_stride = 5
loss_stride = 5
gradient_stride = 5

# File paths
files_image = {
    'Vanilla': '../image/results/stability_metrics_vanilla.json',
    'Batch Norm': '../image/results/stability_metrics_bn.json',
    'NP-reg': '../image/results/stability_metrics_np.json',
}

files_tabular = {
    'Vanilla': '../tabular/results/stability_metrics_vanilla.json',
    'Batch Norm': '../tabular/results/stability_metrics_bn.json',
    'NP-reg': '../tabular/results/stability_metrics_np.json',
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
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

def load_metrics(files_dict):
    metrics = {}
    for label, filepath in files_dict.items():
        with open(filepath) as f:
            data_dict = json.load(f)
        
        loss_landscapes = data_dict.get('loss_landscapes', [])
        gradient_stabilities = data_dict.get('gradient_stabilities', [])
        effective_beta_smoothness = data_dict.get('effective_betas', [])

        loss_mins, loss_maxs = to_min_max(loss_landscapes)
        gradient_mins, gradient_maxs = to_min_max(gradient_stabilities)

        metrics[label] = {
            'loss_maxs': loss_maxs,
            'loss_mins': loss_mins,
            'gradient_maxs': gradient_maxs,
            'gradient_mins': gradient_mins,
            'effective_beta_smoothness': effective_beta_smoothness,
        }
    return metrics

def apply_legend_order(ax, order, **legend_kwargs):
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {label: handle for handle, label in zip(handles, labels)}
    ordered_handles = [label_to_handle[label] for label in order if label in label_to_handle]
    ordered_labels = [label for label in order if label in label_to_handle]
    ax.legend(ordered_handles, ordered_labels, **legend_kwargs)

# Load data
metrics_image = load_metrics(files_image)
metrics_tabular = load_metrics(files_tabular)

# Create 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))

# ==========================================
# 3. ROW 1: IMAGE DATASETS (With Titles)
# ==========================================
x_tick_step_row1 = 400

# (a) Loss landscape variation
ax = axes[0, 0]
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x*100)}'))
for label in ['NP-reg', 'Vanilla', 'Batch Norm']:
    steps = np.arange(len(metrics_image[label]['loss_maxs']))[::loss_stride]
    loss_mins = metrics_image[label]['loss_mins'][::loss_stride]
    loss_maxs = metrics_image[label]['loss_maxs'][::loss_stride]
    ax.fill_between(steps, loss_mins, loss_maxs, color=colors[label], label=label, alpha=0.7, linewidth=1.5)

ax.set_xlabel('Steps')
ax.set_ylabel('Loss Landscape')
max_steps = max(len(metrics_image[label]['loss_maxs']) for label in metrics_image)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row1))
ax.set_xlim(0, max_steps - 1)
ax.set_title('(a)')
ax.set_yscale('log')
ax.set_yticks([1, 3])
ax.set_ylim(0.6, 5)
ax.set_yticklabels([r'$10^0$', r'$3\times10^0$'])
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)

# (b) Gradient predictiveness
ax = axes[0, 1]
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x*100)}'))
z_orders_r1_b = {'Batch Norm': 3, 'NP-reg': 1, 'Vanilla': 2}
for label in ['Vanilla', 'NP-reg', 'Batch Norm']:
    steps = np.arange(len(metrics_image[label]['gradient_maxs']))[::gradient_stride]
    gradient_mins = metrics_image[label]['gradient_mins'][::gradient_stride]
    gradient_maxs = metrics_image[label]['gradient_maxs'][::gradient_stride]
    ax.fill_between(steps, gradient_mins, gradient_maxs, color=colors[label], label=label, alpha=0.7, linewidth=1.5, zorder=z_orders_r1_b[label])

ax.set_xlabel('Steps')
ax.set_ylabel('Gradient Predictiveness')
ax.set_title('(b)')
max_steps = max(len(metrics_image[label]['gradient_maxs']) for label in metrics_image)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row1))
ax.set_xlim(0, max_steps - 1)
ax.set_ylim(0, 18)
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)

# (c) Effective beta smoothness
ax = axes[0, 2]
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x*100)}'))
for label in ['NP-reg', 'Vanilla', 'Batch Norm']:
    data_m_full = metrics_image[label]['effective_beta_smoothness']
    data_m = data_m_full[::effective_beta_stride]
    steps = np.arange(len(data_m_full))[::effective_beta_stride]
    sns.lineplot(x=steps, y=data_m, ax=ax, color=colors[label], linewidth=1.5, label=label, alpha=0.7)

ax.set_xlabel('Steps')
ax.set_ylabel(r'$\beta$-smoothness')
ax.set_title('(c)')
max_steps = max(len(metrics_image[label]['effective_beta_smoothness']) for label in metrics_image)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row1))
ax.set_xlim(0, max_steps - 1)
ax.set_ylim(0, 70)
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)


# ==========================================
# 4. ROW 2: TABULAR / HAR DATASETS (No Titles)
# ==========================================
x_tick_step_row2 = 600

# Loss landscape variation
ax = axes[1, 0]
z_orders_r2_a = {'Batch Norm': 2, 'NP-reg': 3, 'Vanilla': 1}
for label in ['NP-reg', 'Vanilla', 'Batch Norm']:
    steps = np.arange(len(metrics_tabular[label]['loss_maxs']))[::loss_stride]
    loss_mins = metrics_tabular[label]['loss_mins'][::loss_stride]
    loss_maxs = metrics_tabular[label]['loss_maxs'][::loss_stride]
    ax.fill_between(steps, loss_mins, loss_maxs, color=colors[label], label=label, alpha=0.7, linewidth=1.5, zorder=z_orders_r2_a[label])

ax.set_xlabel('Steps')
ax.set_ylabel('Loss Landscapes')
max_steps = max(len(metrics_tabular[label]['loss_maxs']) for label in metrics_tabular)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row2))
ax.set_xlim(0, max_steps - 1)
ax.set_yscale('log')
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)

# Gradient predictiveness
ax = axes[1, 1]
z_orders_r2_b = {'Batch Norm': 1, 'NP-reg': 3, 'Vanilla': 2}
for label in ['Vanilla', 'NP-reg', 'Batch Norm']:
    steps = np.arange(len(metrics_tabular[label]['gradient_maxs']))[::gradient_stride]
    gradient_mins = metrics_tabular[label]['gradient_mins'][::gradient_stride]
    gradient_maxs = metrics_tabular[label]['gradient_maxs'][::gradient_stride]
    ax.fill_between(steps, gradient_mins, gradient_maxs, color=colors[label], label=label, alpha=0.7, linewidth=1.5, zorder=z_orders_r2_b[label])

ax.set_xlabel('Steps')
ax.set_ylabel('Gradient Predictiveness')
max_steps = max(len(metrics_tabular[label]['gradient_maxs']) for label in metrics_tabular)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row2))
ax.set_xlim(0, max_steps - 1)
ax.set_ylim(0, 4)
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)

# Effective beta smoothness
ax = axes[1, 2]
for label in ['NP-reg', 'Vanilla', 'Batch Norm']:
    data_m_full = metrics_tabular[label]['effective_beta_smoothness']
    data_m = data_m_full[::effective_beta_stride]
    steps = np.arange(len(data_m_full))[::effective_beta_stride]
    sns.lineplot(x=steps, y=data_m, ax=ax, color=colors[label], linewidth=1.5, label=label, alpha=0.7)

ax.set_xlabel('Steps')
ax.set_ylabel(r'$\beta$-smoothness')
max_steps = max(len(metrics_tabular[label]['effective_beta_smoothness']) for label in metrics_tabular)
ax.set_xticks(np.arange(0, max_steps + 1, x_tick_step_row2))
ax.set_xlim(0, max_steps - 1)
ax.set_ylim(0, 1000)
apply_legend_order(ax, legend_order, loc='upper right', handlelength=legend_handlelength)


# ==========================================
# 5. SAVE COMBINED OUTPUT
# ==========================================
plt.tight_layout()
plt.savefig('results/combined_stability_metrics_plots.pdf', bbox_inches='tight', pad_inches=0.02)
plt.close()