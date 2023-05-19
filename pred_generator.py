import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Generate binary targets (0 and 1)
targets = np.random.randint(2, size=1000)

# Generate predicted values based on the target distribution
positive_indices = np.where(targets == 1)[0]
negative_indices = np.where(targets == 0)[0]
random_indices = np.random.randint(0, 1001, size=500)

predicted_values = np.zeros_like(targets, dtype=float)
predicted_values[positive_indices] = np.random.uniform(0.5, 1, size=len(positive_indices))
predicted_values[negative_indices] = np.random.uniform(0, 0.5, size=len(negative_indices))
predicted_values[random_indices] = np.random.uniform(0, 1, size=len(random_indices))

# Create a dataframe with the binary targets and predictions
df = pd.DataFrame({'Target': targets, 'Predicted': predicted_values})

df.to_csv('example_preds.csv',index=False) 