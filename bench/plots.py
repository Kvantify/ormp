import pandas as pd
import matplotlib.pyplot as plt

# Accuracy
df = pd.read_csv("ormp-omp.csv", header=None)
df.columns = ["n", "ORMP (mean)", "ORMP (std)", "OMP (mean)", "OMP (std)"]
ax = df.plot(x="n", y="OMP (mean)", marker="*", label="OMP")
ax.fill_between(
    df.n,
    df["OMP (mean)"] - df["OMP (std)"],
    df["OMP (mean)"] + df["OMP (std)"],
    alpha=0.2,
)
df.plot(ax=ax, x="n", y="ORMP (mean)", marker=".", label="ORMP")
ax.fill_between(
    df.n,
    df["ORMP (mean)"] - df["ORMP (std)"],
    df["ORMP (mean)"] + df["ORMP (std)"],
    alpha=0.2,
)
plt.ylabel("Time [s]")
plt.legend(loc="upper left")
plt.savefig("ormp-omp.pdf", dpi=300)
plt.clf()

# Accuracy as function of n
df = pd.read_csv("accuracy-large.csv", header=None)
df.columns = ["n", "m", "k", "OMP wins", "ORMP wins", "Draw"]
ax = df.plot(x="n", y="OMP wins", marker="*")
df.plot(x="n", y="ORMP wins", marker=".", ax=ax)
df.plot(x="n", y="Draw", marker=".", ax=ax)
plt.ylabel("Number of wins out of 10,000 runs")
plt.savefig("ormp-omp-wins.pdf", dpi=300)
plt.clf()

# Accuracy as function of k
df = pd.read_csv("accuracy-various-nonzero.csv", header=None)
df.columns = ["n", "m", "k", "OMP wins", "ORMP wins", "Draw"]
ax = df.plot(x="k", y="OMP wins", marker="*")
df.plot(x="k", y="ORMP wins", marker=".", ax=ax)
df.plot(x="k", y="Draw", marker=".", ax=ax)
plt.ylabel("Number of wins out of 10,000 runs")
plt.savefig("ormp-omp-nonzeros.pdf", dpi=300)
plt.clf()

# Win ratio plot
dfs = [pd.read_csv(f"accuracy-large-{i}.csv", header=None) for i in range(1, 6)]
for df in dfs:
    df.columns = ["n", "m", "k", "OMP wins", "ORMP wins", "Draw"]
    df["Ratio"] = df["ORMP wins"] / df["OMP wins"]
fig, axs = plt.subplots(1, 1)
plt.ylabel("ORMP wins / OMP wins")
for i, df in enumerate(dfs):
    df.plot(x="n", y="Ratio", ax=axs, label=f"g = {i+1}", marker=".")
plt.plot((0, 1000), (1, 1), "--")
plt.savefig("ormp-omp-win-ratio.pdf", dpi=300)
plt.clf()
