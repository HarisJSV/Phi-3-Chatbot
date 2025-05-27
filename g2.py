# Define the evaluation metrics
rouge_metrics = ['First Set', 'Second Set', 'Third Set', 'Fourth Set', 'Fifth Set']
rouge1_values = [0.2278, 0.2687, 0.4722, 0.4691, 0.6154]  # ROUGE-1 values
rouge2_values = [0.1039, 0.0923, 0.2857, 0.3038, 0.4800]  # ROUGE-2 values

# Set the bar width
bar_width = 0.35

# Set the x locations for the groups
r1 = np.arange(len(rouge_metrics))
r2 = [x + bar_width for x in r1]

# Create the bars
plt.bar(r1, rouge1_values, width=bar_width, color='blue', edgecolor='grey', label='ROUGE-1')
plt.bar(r2, rouge2_values, width=bar_width, color='orange', edgecolor='grey', label='ROUGE-2')

# Add labels and title
plt.xlabel('Evaluation Sets', fontweight='bold')
plt.xticks([r + bar_width / 2 for r in range(len(rouge_metrics))], rouge_metrics)
plt.ylabel('Score', fontweight='bold')
plt.title('ROUGE-1 and ROUGE-2 Scores')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
