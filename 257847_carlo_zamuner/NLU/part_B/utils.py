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

# Importiamo il tokenizer di BERT (ora anche la versione Fast esplicitamente)
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
)  # <-- Importato anche BertTokenizerFast


# --- Variabili Globali ---

# Determina il dispositivo (MPS per Mac M1/M2/M3, CUDA per NVIDIA, o CPU)
if torch.backends.mps.is_available():  # Controlla prima per MPS
    device = torch.device("mps")
elif torch.cuda.is_available():  # Poi controlla per CUDA
    device = torch.device("cuda:0")  # Usa la prima GPU disponibile
else:  # Altrimenti usa la CPU
    device = torch.device("cpu")

print(f"Dispositivo selezionato per PyTorch: {device}")

if device.type == "cuda":
    # Usato per reportare errori specifici sul lato CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# L'indice PAD_TOKEN è 0, che è anche l'ID del token [PAD] per la maggior parte dei tokenizer di BERT (incluso bert-base-uncased)
PAD_TOKEN_ID = 0  # Sarà sovrascritto dopo aver inizializzato il tokenizer
IGNORE_INDEX = -100  # Indice speciale per le etichette di slot da IGNORARE


# --- Caricamento Dati ---
def load_data(path):
    """
    input: path/to/data (percorso del file JSON)
    output: lista di dizionari (il dataset caricato)
    """
    dataset = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
    except FileNotFoundError:
        print(f"Errore: File dati non trovato in {path}")
        print(
            "Assicurati che i file del dataset ATIS (train.json, test.json) siano nella directory 'dataset/ATIS'."
        )
        print("Potrebbe essere necessario scaricarli.")
        exit()
    except json.JSONDecodeError:
        print(
            f"Errore: Impossibile decodificare JSON da {path}. Controlla il formato del file."
        )
        exit()
    return dataset


# --- Classe Lang (Gestione Vocabolari SOLO per Etichette) ---
class Lang:
    """Classe helper per gestire i mapping tra label di intent/slot e indici numerici.
    Non gestisce il vocabolario delle parole, che è gestito dal tokenizer di BERT."""

    def __init__(self, intents, slots):
        """
        Inizializza i dizionari intent2id, slot2id e i loro inversi.
        Args:
            intents (set): Set di tutte le etichette di intent uniche.
            slots (set): Set di tutte le etichette di slot uniche.
        """
        self.slot2id = self.lab2id(slots, pad=True)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        """Costruisce il dizionario label -> id (per slot e intenti)."""
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN_ID
        for elem in sorted(list(elements)):
            if elem not in vocab:
                vocab[elem] = len(vocab)
        return vocab


# --- Classe Dataset (IntentsAndSlots per BERT) ---
class IntentsAndSlotsBERT(data.Dataset):
    """Classe Dataset custom per PyTorch per gestire il dataset ATIS con tokenizzazione BERT."""

    def __init__(self, dataset, lang, tokenizer, max_len=50):
        """
        Inizializza il dataset, tokenizza con BERT e allinea le etichette di slot.
        Args:
            dataset (list): Lista di dizionari (dati grezzi).
            lang (Lang): Oggetto Lang contenente i vocabolari delle etichette.
            tokenizer (BertTokenizerFast): Tokenizer di BERT Fast.
            max_len (int): Lunghezza massima delle sequenze dopo tokenizzazione BERT.
        """
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang

        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = []
        self.intent_labels = []
        self.slot_labels = []

        self.original_words = []
        self.original_slot_labels = []
        self.word_ids_list = []

        self.ignore_index = IGNORE_INDEX
        self.pad_token_id = tokenizer.pad_token_id

        for item in dataset:
            utterance = item["utterance"]
            original_words = utterance.split()
            original_slot_labels = item["slots"].split()

            # --- Tokenizzazione e Allineamento ---
            # --- MODIFICA: Rimuovi return_word_ids da encode_plus ---
            encoded_inputs = tokenizer.encode_plus(
                utterance,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
                # return_word_ids=True, # <-- Rimosso da qui
            )
            # --- Fine Modifica ---

            input_ids = encoded_inputs["input_ids"].squeeze(0)
            attention_mask = encoded_inputs["attention_mask"].squeeze(0)
            token_type_ids = encoded_inputs["token_type_ids"].squeeze(0)

            # --- MODIFICA: Ottieni word_ids separatamente ---
            # encode_plus restituisce un oggetto BatchEncoding che ha il metodo word_ids()
            word_ids = encoded_inputs.word_ids(batch_index=0)
            # --- Fine Modifica ---

            # Verifica che word_ids sia stato ottenuto correttamente
            if word_ids is None:
                print(
                    f"Warning: word_ids non ottenuto per utterance: {utterance}. L'allineamento potrebbe fallire."
                )
                # Puoi decidere se saltare questo campione o usare un placeholder
                # word_ids = [None] * len(input_ids) # Placeholder (ma non ideale)
                continue  # Salta questo campione se word_ids non è disponibile

            # --- Gestione Etichette Slot con Allineamento Sub-token ---
            aligned_slot_labels = [self.ignore_index] * len(input_ids)
            last_word_id = None
            for token_idx, word_id in enumerate(word_ids):
                if token_idx >= len(input_ids):
                    break
                if word_id is not None:
                    is_first_subtoken = last_word_id is None or word_id != last_word_id
                    if is_first_subtoken:
                        if word_id < len(original_slot_labels):
                            original_slot_label = original_slot_labels[word_id]
                            slot_label_id = self.lang.slot2id.get(
                                original_slot_label, self.ignore_index
                            )
                            aligned_slot_labels[token_idx] = slot_label_id
                        else:
                            aligned_slot_labels[token_idx] = self.ignore_index
                if word_id is not None:
                    last_word_id = word_id
                else:
                    last_word_id = None
            assert len(aligned_slot_labels) == len(input_ids)

            # --- Gestione Etichetta Intent ---
            intent_label_id = self.lang.intent2id.get(item["intent"], self.ignore_index)

            # Salva i dati processati
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.token_type_ids.append(token_type_ids)
            self.intent_labels.append(intent_label_id)
            self.slot_labels.append(torch.LongTensor(aligned_slot_labels))
            self.original_words.append(original_words)
            self.original_slot_labels.append(original_slot_labels)
            self.word_ids_list.append(word_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Restituisce i dati tokenizzati e allineati come tensori, più le info originali per eval
        sample = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "token_type_ids": self.token_type_ids[idx],
            "intent_labels": self.intent_labels[idx],
            "slot_labels": self.slot_labels[idx],
            "original_words": self.original_words[idx],
            "original_slot_labels": self.original_slot_labels[idx],
            "word_ids": self.word_ids_list[idx],
        }
        return sample


# --- Funzione Collate (per DataLoader con input BERT) ---
def collate_fn_bert(data):
    """
    Funzione custom per creare un batch da campioni tokenizzati BERT.
    Gestisce lo stacking dei tensori e li sposta sul device.
    Raccoglie anche le liste di info originali e word_ids (non le stacka).
    """
    input_ids = torch.stack([d["input_ids"] for d in data])
    attention_masks = torch.stack([d["attention_mask"] for d in data])
    token_type_ids = torch.stack([d["token_type_ids"] for d in data])
    intent_labels = torch.LongTensor([d["intent_labels"] for d in data])
    slot_labels = torch.stack([d["slot_labels"] for d in data])
    original_words_batch = [d["original_words"] for d in data]
    original_slot_labels_batch = [d["original_slot_labels"] for d in data]
    word_ids_batch = [d["word_ids"] for d in data]

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    token_type_ids = token_type_ids.to(device)
    intent_labels = intent_labels.to(device)
    slot_labels = slot_labels.to(device)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "token_type_ids": token_type_ids,
        "intent_labels": intent_labels,
        "slot_labels": slot_labels,
        "original_words": original_words_batch,
        "original_slot_labels": original_slot_labels_batch,
        "word_ids": word_ids_batch,
    }
    return batch


# --- Inizializzazione Pesi (Opzionale per i nuovi layer) ---
def init_weights_linear(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# --- END OF FILE utils.py ---
