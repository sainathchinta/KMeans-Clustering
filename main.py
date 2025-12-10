import numpy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
print("Current folder:", os.getcwd())

data = pd.read_csv("/Users/sai/Documents/k-means/faithful.csv")


# get column names

print("column names : " + data.columns)

# Get number of rows and columns
print(data.shape)

# Get basic statistics (mean, min, max, etc.)
print(data.describe())


# scatter plot

plt.plot(
    data["eruptions"],
    data["waiting"],
    linestyle="None",      # ❗ no line
    marker="o",            # circle marker
    markersize=8,          # dot size
    markerfacecolor="blue",# inside color
    markeredgecolor="black",# border color
    markeredgewidth=1
)
# Add labels and title
plt.xlabel('Eruption duration (minutes)')
plt.ylabel('Waiting time to next eruption (minutes)')
plt.title('Old Faithful Geyser Data')

# ✅ SAVE BEFORE SHOW
plt.savefig("faithful_correct.png")

# ✅ THEN show
plt.show()