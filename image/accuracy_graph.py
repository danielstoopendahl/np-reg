import matplotlib.pyplot as plt
import numpy as np

# Data
model_sizes = [4096, 8192, 16384]
techniques = ['Vanilla', 'Weight Decay', 'Dropout', 'Batch Norm', 'NP-reg', 'O-reg']

averages = {
    'Vanilla': [59.38, 59.50, 60.08],
    'Weight Decay': [59.77, 59.91, 60.18],
    'Dropout': [59.73, 59.35, 59.59],
    'Batch Norm': [66.52, 67.10, 67.40],
    'NP-reg': [65.38, 65.71, 66.25],
    'O-reg': [65.14, 66.88, 67.64]
}

se = {
    'Vanilla': [0.12, 0.17, 0.18],
    'Weight Decay': [0.18, 0.21, 0.18],
    'Dropout': [0.17, 0.08, 0.22],
    'Batch Norm': [0.22, 0.16, 0.18],
    'NP-reg': [0.13, 0.13, 0.21],
    'O-reg': [0.18, 0.19, 0.32]
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
plt.xticks(model_sizes) # Ensure ticks match model sizes
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('accuracy_plot.png')