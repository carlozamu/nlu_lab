import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Cartella contenente i risultati (modifica se necessario)
results_dir = "results"

# Lista per raccogliere i record da ogni configurazione
records = []

# Pattern per estrarre i valori di emb_dropout, fc_dropout e alpha dal nome della cartella
pattern = r"emb([\d\.]+)_fc([\d\.]+)_alpha([\d\.]+)"

# Scansiona ogni sottocartella in results
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):
        csv_file = os.path.join(folder_path, "training_summary.csv")
        if os.path.exists(csv_file):
            # Legge il CSV per ottenere il Dev PPL minimo
            df = pd.read_csv(csv_file)
            # Supponiamo che il file CSV abbia una colonna chiamata "Dev PPL"
            min_dev_ppl = df["Dev PPL"].min()
            
            # Estrae gli iperparametri dal nome della cartella
            m = re.match(pattern, folder)
            if m:
                emb_dp = float(m.group(1))
                fc_dp = float(m.group(2))
                alpha = float(m.group(3))
            else:
                emb_dp = fc_dp = alpha = None

            records.append({
                "Configuration": folder,
                "emb_dropout": emb_dp,
                "fc_dropout": fc_dp,
                "alpha": alpha,
                "min_dev_ppl": min_dev_ppl
            })

# Crea un DataFrame dai dati raccolti
results_df = pd.DataFrame(records)
results_df = results_df.sort_values("min_dev_ppl")

# Salva i risultati aggregati in un file CSV (opzionale)
results_df.to_csv("comparison_results.csv", index=False)
print("Aggregated results saved in comparison_results.csv")

# ---------------------------
# Bar Plot: Min Dev PPL per Configurazione
# ---------------------------
plt.figure(figsize=(12, 6))
plt.bar(range(len(results_df)), results_df["min_dev_ppl"], color="skyblue")
plt.xticks(range(len(results_df)), results_df["Configuration"], rotation=90)
plt.title("Min Dev PPL per Configuration")
plt.ylabel("Min Dev PPL")
plt.xlabel("Configuration")
plt.tight_layout()
plt.savefig("barplot_min_dev_ppl.png")
plt.close()
print("Bar plot saved as barplot_min_dev_ppl.png")

# ---------------------------
# Scatter Plot: emb_dropout vs min_dev_ppl, color-coded by fc_dropout
# ---------------------------
plt.figure(figsize=(10, 6))
unique_fc = sorted(results_df["fc_dropout"].unique())
for fc_dp in unique_fc:
    subset = results_df[results_df["fc_dropout"] == fc_dp]
    plt.scatter(subset["emb_dropout"], subset["min_dev_ppl"], label=f"fc_dropout = {fc_dp}", s=100)

plt.xlabel("emb_dropout")
plt.ylabel("Min Dev PPL")
plt.title("Min Dev PPL vs emb_dropout (grouped by fc_dropout)")
plt.legend(title="fc_dropout")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()
print("Scatter plot saved as scatter_plot.png")