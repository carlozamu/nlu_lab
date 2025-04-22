# --- START OF FILE utils.py ---

import json
import os
import random
from collections import Counter
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --- Variabili Globali ---

# Determina il dispositivo da usare: MPS (Apple Silicon), CUDA (NVIDIA GPU), o CPU
if torch.backends.mps.is_available():  # Controlla prima per MPS (Mac M1/M2/M3...)
    device = torch.device("mps")
elif torch.cuda.is_available():  # Poi controlla per CUDA (GPU NVIDIA)
    device = torch.device("cuda:0")  # Usa la prima GPU disponibile
else:  # Altrimenti usa la CPU
    device = torch.device("cpu")

print(f"Dispositivo selezionato per PyTorch: {device}")

if device.type == "cuda":
    # Usato per reportare errori specifici sul lato CUDA (utile per debug)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Indice speciale per il token di padding
PAD_TOKEN = 0


# --- Caricamento Dati ---
def load_data(path):
    """
    input: path/to/data (percorso del file JSON)
    output: lista di dizionari (il dataset caricato)
    """
    dataset = []
    try:
        # Apre e legge il file JSON
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
    except FileNotFoundError:
        # Gestisce l'errore se il file non viene trovato
        print(f"Errore: File dati non trovato in {path}")
        print(
            "Assicurati che i file del dataset ATIS (train.json, test.json) siano nella directory 'dataset/ATIS'."
        )
        print("Potrebbe essere necessario scaricarli.")
        exit()  # Esce dallo script se i dati mancano
    except json.JSONDecodeError:
        # Gestisce l'errore se il JSON non è valido
        print(
            f"Errore: Impossibile decodificare JSON da {path}. Controlla il formato del file."
        )
        exit()
    return dataset


# --- Classe Lang (Gestione Vocabolari) ---
class Lang:
    """Classe helper per gestire i mapping tra parole/label e indici numerici."""

    def __init__(self, words, intents, slots, cutoff=0):
        """
        Inizializza i dizionari word2id, slot2id, intent2id e i loro inversi.
        Args:
            words (list): Lista di tutte le parole nel training set (per il vocabolario).
            intents (set): Set di tutte le etichette di intent uniche.
            slots (set): Set di tutte le etichette di slot uniche.
            cutoff (int): Frequenza minima per includere una parola nel vocabolario (0 per includere tutto).
        """
        # Crea il mapping parola -> id, applicando opzionalmente un cutoff e aggiungendo 'unk'
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        # Crea il mapping slot -> id (include 'pad')
        self.slot2id = self.lab2id(slots, pad=True)
        # Crea il mapping intent -> id (non include 'pad')
        self.intent2id = self.lab2id(intents, pad=False)
        # Crea i mapping inversi id -> parola/label
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        """Costruisce il dizionario parola -> id."""
        # Inizia con il token 'pad'
        vocab = {"pad": PAD_TOKEN}
        # Aggiunge il token 'unk' (sconosciuto) se richiesto
        if unk:
            vocab["unk"] = len(vocab)
        # Conta la frequenza delle parole
        count = Counter(elements)
        # Aggiunge parole al vocabolario se la loro frequenza supera il cutoff
        for k, v in count.items():
            if cutoff is None or v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        """Costruisce il dizionario label -> id (per slot e intenti)."""
        vocab = {}
        # Aggiunge il token 'pad' se richiesto (tipicamente per gli slot)
        if pad:
            vocab["pad"] = PAD_TOKEN
        # Ordina gli elementi per assicurare un mapping consistente tra diverse esecuzioni
        for elem in sorted(list(elements)):
            # Assegna un id univoco a ciascuna etichetta
            vocab[elem] = len(vocab)
        return vocab


# --- Classe Dataset (IntentsAndSlots) ---
class IntentsAndSlots(data.Dataset):
    """Classe Dataset custom per PyTorch per gestire il dataset ATIS."""

    # Metodi obbligatori: __init__, __len__, __getitem__
    def __init__(self, dataset, lang, unk="unk"):
        """
        Inizializza il dataset, converte testi e label in sequenze di indici.
        Args:
            dataset (list): Lista di dizionari (dati grezzi caricati).
            lang (Lang): Oggetto Lang contenente i vocabolari.
            unk (str): Token da usare per le parole fuori vocabolario.
        """
        self.utterances = []  # Lista delle frasi (stringhe)
        self.intents = []  # Lista degli intenti (stringhe)
        self.slots = []  # Lista delle sequenze di slot (stringhe)
        self.unk = unk  # Token per parole sconosciute

        # Popola le liste con i dati grezzi
        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        # Converte le sequenze testuali in sequenze di indici usando i vocabolari
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        """Restituisce la dimensione totale del dataset."""
        return len(self.utterances)

    def __getitem__(self, idx):
        """
        Restituisce un singolo campione dal dataset all'indice specificato.
        Args:
            idx (int): Indice del campione da recuperare.
        Returns:
            dict: Dizionario contenente 'utterance', 'slots' (come LongTensor) e 'intent' (come int).
        """
        # Converte le liste di indici in Tensori PyTorch di tipo Long (necessario per Embedding e Loss)
        utt = torch.LongTensor(self.utt_ids[idx])
        slots = torch.LongTensor(self.slot_ids[idx])
        # L'intent è un singolo indice (int), non serve convertirlo qui
        intent = self.intent_ids[idx]
        # Restituisce il campione come dizionario, la struttura attesa da collate_fn
        sample = {"utterance": utt, "slots": slots, "intent": intent}
        return sample

    # --- Metodi Ausiliari per il Mapping ---
    def mapping_lab(self, data, mapper):
        """Mappa una lista di etichette (intenti) ai loro indici."""
        # Se l'etichetta è nel mapper usa il suo id, altrimenti usa l'id di 'unk' (se esiste)
        # Gestisce il caso in cui 'unk' non sia nel mapper (non dovrebbe succedere per intenti/slot)
        return [
            mapper.get(x, mapper.get(self.unk))
            for x in data
            if x in mapper or self.unk in mapper
        ]

    def mapping_seq(self, data, mapper):
        """Mappa una lista di sequenze (utterances, slots) a liste di indici."""
        res = []
        for seq in data:
            tmp_seq = []
            # Assicura che seq sia trattato come stringa prima di splittare
            for x in str(seq).split():
                # Mappa ogni token all'id corrispondente, o all'id 'unk' se non presente
                # Usa .get(self.unk, PAD_TOKEN) come fallback se 'unk' non fosse definito
                tmp_seq.append(mapper.get(x, mapper.get(self.unk, PAD_TOKEN)))
            res.append(tmp_seq)
        return res


# --- Funzione Collate (per DataLoader) ---
def collate_fn(data):
    """
    Funzione custom per processare una lista di campioni (restituiti da __getitem__)
    e creare un batch. Ordina le sequenze per lunghezza, applica padding,
    e sposta i tensori sul dispositivo corretto (CPU/GPU/MPS).
    Args:
        data (list): Lista di dizionari, output di __getitem__ per un batch.
    Returns:
        dict: Un batch pronto per essere dato in input al modello.
    """

    def merge(sequences):
        """
        Funzione interna per fare il padding di un batch di sequenze.
        Args:
            sequences (list): Lista di tensori (sequenze di indici).
        Returns:
            tuple: (padded_seqs, lengths)
                   - padded_seqs: Tensore Long con le sequenze paddate (batch_size, max_len).
                   - lengths: Tensore Long con le lunghezze originali (batch_size).
        """
        # Calcola le lunghezze di ogni sequenza nel batch
        lengths = [len(seq) for seq in sequences]
        # Trova la lunghezza massima nel batch (gestisce batch vuoti)
        max_len = max(lengths) if lengths else 0
        # Crea un tensore pieno di PAD_TOKEN (0) della dimensione giusta
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        # Copia i dati di ogni sequenza nel tensore paddato
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        # Restituisce le sequenze paddate e le loro lunghezze originali come tensore
        return padded_seqs, torch.LongTensor(lengths)

    # Ordina i dati nel batch per lunghezza decrescente dell'utterance
    data.sort(key=lambda x: len(x["utterance"]), reverse=True)
    # Raggruppa i dati per chiave ('utterance', 'slots', 'intent')
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Applica la funzione merge (padding) alle sequenze di utterance e slot
    src_utt, utt_lengths = merge(new_item["utterance"])
    y_slots, _ = merge(
        new_item["slots"]
    )  # Le lunghezze degli slot sono uguali a utt_lengths

    # Raccoglie gli intenti (che sono singoli indici) in un unico tensore Long
    intent = torch.LongTensor(new_item["intent"])

    # Sposta tutti i tensori sul dispositivo designato (definito globalmente)
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    # Le lunghezze sono necessarie sulla CPU per pack_padded_sequence,
    # ma le spostiamo sul device come nel codice originale per consistenza.
    # Nota: pack_padded_sequence internamente le userà dalla CPU se necessario.
    utt_lengths = utt_lengths.to(device)  # Mantenuto come nell'originale

    # Crea il dizionario finale del batch con chiavi consistenti
    batch = {
        "utterances": src_utt,  # Sequenze di input paddate
        "intents": intent,  # Etichette di intent
        "y_slots": y_slots,  # Etichette di slot paddate (ground truth)
        "slots_len": utt_lengths,  # Lunghezze originali (usate per packing e calcolo loss/metriche)
    }
    return batch


# --- Inizializzazione Pesi ---
def init_weights(mat):
    """Inizializza i pesi del modello secondo strategie comuni."""
    # Itera su tutti i moduli (layer) del modello
    for m in mat.modules():
        # Se il modulo è un tipo di RNN (GRU, LSTM, RNN)
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            # Itera su tutti i parametri del modulo RNN
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    # Inizializza i pesi input-hidden con Xavier Uniform
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    # Inizializza i pesi hidden-hidden con Orthogonal
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    # Inizializza i bias a zero
                    param.data.fill_(0)
        # Se il modulo è un layer Lineare (fully connected)
        elif type(m) in [nn.Linear]:
            # Inizializza i pesi con una distribuzione Uniforme piccola
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            # Se esiste il bias
            if m.bias is not None:
                # Inizializza il bias a un piccolo valore costante (es. 0.01)
                m.bias.data.fill_(0.01)


# --- END OF FILE utils.py ---
