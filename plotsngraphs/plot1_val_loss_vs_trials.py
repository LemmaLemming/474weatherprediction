import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("bayes_trials_log.csv")

# Plot: Validation Loss vs Trial Number
plt.figure()
plt.plot(df['trial'], df['val_loss'], marker='o')
plt.title("Validation MSE over Trials")
plt.xlabel("Trial #")
plt.ylabel("Validation MSE")
plt.grid(True)
plt.tight_layout()
plt.show()
