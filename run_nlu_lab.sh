#!/bin/bash

# ==============================================================================
# SBATCH Configuration - MODIFICA QUESTE RIGHE SECONDO LE TUE ESIGENZE/PERMESSI
# ==============================================================================

# Richiedi una partizione specifica (ESEMPIO: 'cpu' o 'edu-thesis' o 'long')
# DEVI SAPERE QUALE PARTIZIONE PUOI USARE! Chiedi al supporto o controlla la doc.
#SBATCH -p edu-long           # <--- CAMBIA QUESTO!

# Imposta un limite di tempo (HH:MM:SS). Inizia con un tempo breve per test.
#SBATCH -t 0-05:00:00     # <--- CAMBIA QUESTO (es. 02:00:00 per 2 ore)

# Richiedi risorse GPU (se il tuo main.py le usa)
# Se NON usi GPU, COMMENTA o RIMUOVI la riga seguente
#SBATCH --gres=gpu:1      # <--- MODIFICA o RIMUOVI se non usi GPU

# Richiedi CPU cores
#SBATCH --cpus-per-task=4 # <--- CAMBIA QUESTO se main.py necessita più/meno CPU

# Altre configurazioni standard (di solito non serve modificarle)
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --job-name=nlu_lab_main # Nome del job per squeue
#SBATCH --mem-per-cpu=4G
# Percorsi per i file di output e di errore dello script SLURM
# ASSICURATI CHE QUESTE DIRECTORY ESISTANO e siano scrivibili!
# Puoi creare una cartella 'slurm_output' nel tuo progetto.
# mkdir -p /home/carlo.zamuner/nlu_lab/slurm_output
#SBATCH -o /home/carlo.zamuner/nlu_lab/slurm_output/main_py_%j.out  # %j aggiunge l'ID del job al nome file
#SBATCH -e /home/carlo.zamuner/nlu_lab/slurm_output/main_py_%j.err  # %j aggiunge l'ID del job al nome file

# ==============================================================================
# Setup Ambiente Conda
# ==============================================================================
echo "INFO: Configurazione ambiente Conda..."

# Assicurati che il percorso a Miniconda/Anaconda sia corretto
export PATH="/home/carlo.zamuner/miniconda3/bin:$PATH"

# Inizializza Conda per la shell Bash
eval "$(conda shell.bash hook)"

# Attiva il tuo ambiente Conda specifico
conda activate nlu_env # <--- Assicurati che sia il nome corretto!
if [ $? -ne 0 ]; then
    echo "ERRORE: Impossibile attivare l'ambiente Conda 'nlu_env'"
    exit 1
fi
echo "INFO: Ambiente Conda 'nlu_env' attivato."
echo "INFO: Versione Python nell'ambiente:"
python --version

# ==============================================================================
# Caricamento Moduli Aggiuntivi (se necessario)
# ==============================================================================

# Carica il modulo CUDA se hai richiesto una GPU (--gres=gpu:1)
# Se non usi GPU, commenta o rimuovi questa sezione
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "INFO: Caricamento modulo CUDA..."
  module load cuda # Carica la versione default (12.4)
  # Se serve una versione specifica: module load cuda/12.5
  if [ $? -ne 0 ]; then
      echo "ERRORE: Impossibile caricare il modulo CUDA"
      exit 1
  fi
  echo "INFO: Modulo CUDA caricato."
  nvidia-smi # Stampa info sulla GPU assegnata (utile per debug)
fi

# ==============================================================================
# Esecuzione dello Script Python Principale
# ==============================================================================
echo "INFO: Esecuzione di main.py..."

# Spostati nella directory dove si trova main.py (se necessario)
# Assumendo che main.py sia in /home/carlo.zamuner/nlu_lab
cd /home/carlo.zamuner/nlu_lab/257847_carlo_zamuner/LM/part_A 

# Esegui il tuo script python
# La parte PROTOCOL_BUFFERS... aiuta a prevenire errori con alcune librerie
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python main.py

# Controlla l'esito dell'esecuzione
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "INFO: Esecuzione di main.py completata con successo."
else
  echo "ERRORE: Esecuzione di main.py fallita con codice di uscita: $EXIT_CODE"
fi

# ==============================================================================
# Pulizia (Disattivazione Conda) - Opzionale ma buona pratica
# ==============================================================================
echo "INFO: Disattivazione ambiente Conda..."
conda deactivate

echo "INFO: Script SLURM terminato."

exit $EXIT_CODE