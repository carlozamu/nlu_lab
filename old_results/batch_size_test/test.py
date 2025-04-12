import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Prova questi percorsi
possible_paths = [
    "optimization_results.csv",
    "./results/optimization_results.csv",
    "./results/temp/optimization_results.csv",
    "./temp/optimization_results.csv",
]

# Trova il primo file disponibile
for path in possible_paths:
    if os.path.exists(path):
        results_path = path
        break
else:
    raise FileNotFoundError(
        "optimization_results.csv non trovato in nessuna delle directory previste."
    )

# Carica i dati
df = pd.read_csv(results_path)

# Assicura i tipi corretti
df["Learning Rate"] = df["Learning Rate"].astype(float)
df["Batch Size"] = df["Batch Size"].astype(int)
df["Best Dev PPL"] = df["Best Dev PPL"].astype(float)

# Trova il miglior risultato
best_result = df.loc[df["Best Dev PPL"].idxmin()]

# Crea una pivot table
pivot = df.pivot_table(
    index="Batch Size", columns="Learning Rate", values="Best Dev PPL", aggfunc="min"
)

# Crea heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot, annot=True, fmt=".1f", cmap="viridis", cbar_kws={"label": "Best Dev PPL"}
)
plt.title("Heatmap of Best Dev PPL\n(Batch Size vs Learning Rate)")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")

# Evidenzia la combinazione migliore
best_x = list(pivot.columns).index(best_result["Learning Rate"])
best_y = list(pivot.index).index(best_result["Batch Size"])
plt.scatter(
    best_x + 0.5,
    best_y + 0.5,
    s=200,
    edgecolors="white",
    facecolors="none",
    linewidths=2,
)

# Salva la figura
output_path = os.path.join(os.path.dirname(results_path), "best_dev_ppl_heatmap.png")
plt.tight_layout()
plt.savefig(output_path)
plt.close()

# Mostra info sul miglior risultato
print("\nMiglior combinazione trovata:")
print(best_result.to_string())
print(f"\nHeatmap salvata in: {output_path}")
