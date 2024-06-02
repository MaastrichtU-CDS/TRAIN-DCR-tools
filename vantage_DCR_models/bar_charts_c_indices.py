import matplotlib.pyplot as plt
import seaborn as sns

# Data
model_types = ['C-Model', 'RP-Model', 'RN-Model', 'CR-Model']
outcomes = ['OS', 'DM', 'LRR']
global_average = {
    'C-Model': [0.62, 0.63, 0.62],
    'RP-Model': [0.52, 0.55, 0.54],
    'RN-Model': [0.50, 0.62, 0.60],
    'CR-Model': [0.57, 0.59, 0.64]
    # 'C-Model': [0.62, 0.64, 0.47],
    # 'RP-Model': [0.53, 0.55, 0.57],
    # 'RN-Model': [0.60, 0.52, 0.64],
    # 'CR-Model': [0.55, 0.56, 0.57]
}

# Define colors

# colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
# colors = ['#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
colors = sns.color_palette("colorblind")

# Grouped bar chart
fig, ax = plt.subplots()
bar_width = 0.2
index = range(len(outcomes))

for i, model_type in enumerate(model_types):
    bars = [global_average[model_type][j] for j in range(len(outcomes))]
    ax.bar([x + i * bar_width for x in index], bars, bar_width, label=model_type, color=colors[i % len(colors)])

ax.set_xlabel('Outcomes')
ax.set_ylabel('Global average')
ax.set_title('C-indices - Loop 1')
ax.set_xticks([x + bar_width * (len(model_types) - 1) / 2 for x in index])
ax.set_xticklabels(outcomes)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("loop1_c_index.png", dpi=300)  # Save the figure as a high-resolution PNG
plt.show()
