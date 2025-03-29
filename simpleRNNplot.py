import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tuning_results.csv")

plt.plot(range(1, len(df) + 1), df['avg_val_mae'], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Average Validation MAE")
plt.title("Bayesian Optimization Convergence")
plt.grid(True)
plt.show()
