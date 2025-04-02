import os
import re
import pandas as pd

# Percorso della cartella results (modifica se necessario)
results_dir = "results"

# Lista per salvare i record per ogni configurazione
records = []

# Pattern per estrarre gli iperparametri dal nome della cartella
pattern = r"emb([\d\.]+)_fc([\d\.]+)_alpha([\d\.]+)"

# Itera su tutte le sottocartelle di results
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path):
        csv_path = os.path.join(folder_path, "training_summary.csv")
        if os.path.exists(csv_path):
            # Legge il CSV
            df = pd.read_csv(csv_path)
            # Calcola il Dev PPL minimo (o altra statistica, ad es. ultimo valore)
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

# Crea un DataFrame dai record raccolti
results_df = pd.DataFrame(records)
# Ordina per Dev PPL minimo (migliore in alto)
results_df = results_df.sort_values("min_dev_ppl")

print("Confronto delle configurazioni (ordinato per Dev PPL minimo):")
print(results_df)

# Se desideri creare una pivot table, ad esempio, per visualizzare l'effetto di emb_dropout e fc_dropout per un certo valore di alpha
# (qui mostriamo la media dei min_dev_ppl per ogni combinazione, raggruppando per emb_dropout e fc_dropout)
pivot = results_df.pivot_table(
    index="emb_dropout", 
    columns="fc_dropout", 
    values="min_dev_ppl", 
    aggfunc="mean"
)
results_df.to_csv("comparison_results.csv", index=False)
print("\nPivot table (Dev PPL minimo medio per emb_dropout vs fc_dropout):")
print(pivot)