import matplotlib.pyplot as plt
import numpy as np

# Define the evaluation metrics
metrics = ['First Set', 'Second Set', 'Third Set', 'Fourth Set', 'Fifth Set']
use_values = [0.6434, 0.5601, 0.5278, 0.6037, 0.7591]  # USE Cosine Similarity values
st_values = [0.8163, 0.7853, 0.8377, 0.6756, 0.8080]  # ST Cosine Similarity values

# Set the bar width
bar_width = 0.35

# Set the x locations for the groups
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]

# Create the bars
plt.bar(r1, use_values, width=bar_width, color='blue', edgecolor='grey', label='USE Cosine Similarity')
plt.bar(r2, st_values, width=bar_width, color='orange', edgecolor='grey', label='ST Cosine Similarity')

# Add labels and title
plt.xlabel('Evaluation Sets', fontweight='bold')
plt.xticks([r + bar_width / 2 for r in range(len(metrics))], metrics)
plt.ylabel('Score', fontweight='bold')
plt.title('USE and ST Cosine Similarity')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
