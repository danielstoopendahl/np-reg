import matplotlib.pyplot as plt
import numpy as np

# Data
model_sizes = [10000, 20000]
techniques = ['Vanilla', 'Weight Decay', 'Dropout', 'Layer Norm', 'GSVD-loss', 'Ortho-loss']

averages = {
    'Vanilla': [87.98, 88.17],
    'Weight Decay': [88.08, 88.12],
    'Dropout': [88.07, 88.30],
    'Layer Norm': [87.77, 88.11],
    'GSVD-loss': [88.30, 88.76],
    'Ortho-loss': [87.86, 87.95]
}

se = {
    'Vanilla': [0.02, 0.02],
    'Weight Decay': [0.02, 0.06],
    'Dropout': [0.02, 0.03],
    'Layer Norm': [0.05, 0.05],
    'GSVD-loss': [0.03, 0.01],
    'Ortho-loss': [0.05, 0.05]
}

# t-value for 95% CI with 4 degrees of freedom (n=5)
t_val = 2.776 

plt.figure(figsize=(10, 6))

for technique in techniques:
    avg = np.array(averages[technique])
    err = t_val * np.array(se[technique])
    plt.errorbar(model_sizes, avg, yerr=err, label=technique, marker='o', capsize=5)

plt.xlabel('Model Size', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy vs Model Size with 95% Confidence Intervals', fontsize=14)
plt.xscale('log', base=2)
plt.xticks(model_sizes, [str(size) for size in model_sizes])
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('accuracy_plot_imdb.png')