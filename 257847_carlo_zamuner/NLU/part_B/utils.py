# Add functions or classes used for data loading and preprocessing
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
# Non useremo pack_padded_sequence/pad_packed_sequence con BERT direttamente, BERT gestisce l'attention mask
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Importiamo il tokenizer di BERT
from transformers import BertTokenizer

# --- Variabili Globali ---

# Determina il dispositivo (MPS per Mac M1/M2/M3, CUDA per NVIDIA, o CPU)
if torch.backends.mps.is_available(): # Controlla prima per MPS
    device = torch.device("mps")
elif torch.cuda.is_available(): # Poi controlla per CUDA
    device = torch.device("cuda:0") # Usa la prima GPU disponibile
else: # Altrimenti usa la CPU
    device = torch.device("cpu")

print(f"Dispositivo selezionato per PyTorch: {device}")

if device.type == 'cuda':
    # Usato per reportare errori specifici sul lato CUDA
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# L'indice PAD_TOKEN è 0, che è anche l'ID del token [PAD] per la maggior parte dei tokenizer di BERT (incluso bert-base-uncased)
# È importante verificare questo per il tokenizer specifico che si usa.
# Per bert-base-uncased: tokenizer.pad_token_id == 0
PAD_TOKEN_ID = 0

# Usiamo -100 come indice speciale per le etichette di slot da IGNORARE nella loss (padding e sub-token non-iniziali)
IGNORE_INDEX = -100


# --- Caricamento Dati ---
def load_data(path):
    '''
        input: path/to/data (percorso del file JSON)
        output: lista di dizionari (il dataset caricato)
    '''
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.loads(f.read())
    except FileNotFoundError:
        print(f"Errore: File dati non trovato in {path}")
        print("Assicurati che i file del dataset ATIS (train.json, test.json) siano nella directory 'dataset/ATIS'.")
        print("Potrebbe essere necessario scaricarli.")
        exit() # Esce dallo script se i dati mancano
    except json.JSONDecodeError:
        print(f"Errore: Impossibile decodificare JSON da {path}. Controlla il formato del file.")
        exit()
    return dataset

# --- Classe Lang (Gestione Vocabolari SOLO per Etichette) ---
class Lang():
    ''' Classe helper per gestire i mapping tra label di intent/slot e indici numerici.
        Non gestisce il vocabolario delle parole, che è gestito dal tokenizer di BERT. '''
    def __init__(self, intents, slots):
        '''
        Inizializza i dizionari intent2id, slot2id e i loro inversi.
        Args:
            intents (set): Set di tutte le etichette di intent uniche.
            slots (set): Set di tutte le etichette di slot uniche.
        '''
        # Non creiamo word2id o id2word qui
        self.slot2id = self.lab2id(slots, pad=True)
        self.intent2id = self.lab2id(intents, pad=False)
        # Non creiamo id2word
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements, pad=True):
        ''' Costruisce il dizionario label -> id (per slot e intenti). '''
        vocab = {}
        # Aggiunge il token 'pad' per gli slot (corrisponderà a IGNORE_INDEX dopo l'allineamento BERT)
        # Usiamo PAD_TOKEN_ID per coerenza visiva nel mapping, ma poi lo convertiremo a IGNORE_INDEX
        if pad:
            vocab['pad'] = PAD_TOKEN_ID
        # Ordina gli elementi per assicurare un mapping consistente tra diverse esecuzioni
        for elem in sorted(list(elements)):
             # Evita di sovrascrivere 'pad' se per caso fosse una label reale (improbabile)
            if elem not in vocab:
                 vocab[elem] = len(vocab)
        return vocab

    # Non serve w2id qui

# --- Classe Dataset (IntentsAndSlots per BERT) ---
class IntentsAndSlotsBERT (data.Dataset):
    ''' Classe Dataset custom per PyTorch per gestire il dataset ATIS con tokenizzazione BERT. '''
    def __init__(self, dataset, lang, tokenizer, max_len=50):
        '''
        Inizializza il dataset, tokenizza con BERT e allinea le etichette di slot.
        Args:
            dataset (list): Lista di dizionari (dati grezzi).
            lang (Lang): Oggetto Lang contenente i vocabolari delle etichette.
            tokenizer (BertTokenizer): Tokenizer di BERT.
            max_len (int): Lunghezza massima delle sequenze dopo tokenizzazione BERT.
        '''
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang

        # Liste per memorizzare i dati tokenizzati e allineati
        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = [] # Utile per coppie di frasi, ma BERT-base lo usa anche per singola (sempre 0)
        self.intent_labels = []
        self.slot_labels = [] # Etichette di slot allineate ai sub-token

        # Indici speciali per BERT (dal tokenizer)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = IGNORE_INDEX # Indice per label slot da ignorare

        # Processa ogni campione nel dataset
        for item in dataset:
            utterance = item['utterance']
            slot_sequence = item['slots'].split() # Splitta la stringa di slot in lista
            intent_label = item['intent']

            # --- Tokenizzazione e Allineamento ---
            # Tokenizza l'utterance. return_offsets_mapping=True è utile per l'allineamento, ma word_ids funziona meglio
            # Add special tokens [CLS] e [SEP] come da paper
            encoded_inputs = tokenizer.encode_plus(
                utterance,
                max_length=self.max_len,
                padding='max_length',      # Applica padding alla lunghezza massima
                truncation=True,           # Tronca se più lungo di max_len
                return_attention_mask=True,# Restituisce l'attention mask
                return_token_type_ids=True,# Restituisce i token type ids
                return_tensors='pt',       # Restituisce tensori PyTorch
                return_word_ids=True       # Richiede gli ID delle parole originali
            )

            input_ids = encoded_inputs['input_ids'].squeeze(0) # Rimuovi la dimensione extra del batch
            attention_mask = encoded_inputs['attention_mask'].squeeze(0)
            token_type_ids = encoded_inputs['token_type_ids'].squeeze(0) # Solitamente tutti 0 per singola sequenza
            word_ids = encoded_inputs.word_ids() # Mappa ID sub-token -> ID parola originale (o None per speciali)

            # --- Gestione Etichette Slot con Allineamento Sub-token ---
            # Crea una lista di etichette slot allineate ai sub-token di BERT
            aligned_slot_labels = []
            last_word_id = None

            for token_idx, word_id in enumerate(word_ids):
                if word_id is None: # Token speciale [CLS], [SEP], [PAD]
                    aligned_slot_labels.append(self.ignore_index)
                elif word_id != last_word_id: # Questo è il PRIMO sub-token di una nuova parola originale
                    # Ottieni l'etichetta slot per la parola originale corrispondente
                    # Gestisci il caso in cui la lista slot_sequence sia più corta del numero di parole (dati corrotti)
                    if word_id < len(slot_sequence):
                         original_slot_label = slot_sequence[word_id]
                         # Mappa l'etichetta stringa al suo ID numerico
                         slot_label_id = self.lang.slot2id.get(original_slot_label, self.ignore_index) # Usa get per UNK o label mancanti
                         aligned_slot_labels.append(slot_label_id)
                    else:
                         # Se word_id è fuori dai limiti di slot_sequence, qualcosa non va
                         print(f"Warning: word_id {word_id} out of bounds for slot_sequence of length {len(slot_sequence)} in utterance: {utterance}")
                         aligned_slot_labels.append(self.ignore_index) # Ignora questa label
                else: # Sub-token successivo della stessa parola originale
                    aligned_slot_labels.append(self.ignore_index) # Ignora questa label nella loss

                last_word_id = word_id

            # Assicurati che la lista allineata abbia la stessa lunghezza di input_ids (dopo padding)
            # encode_plus con padding='max_length' dovrebbe già farlo
            while len(aligned_slot_labels) < self.max_len:
                 aligned_slot_labels.append(self.ignore_index) # Aggiungi IGNORE_INDEX per i token di padding aggiuntivi

            # Tronca le label se necessario (dovrebbe essere gestito da max_len in encode_plus)
            aligned_slot_labels = aligned_slot_labels[:self.max_len]


            # --- Gestione Etichetta Intent ---
            intent_label_id = self.lang.intent2id.get(intent_label, self.ignore_index) # Usa get per UNK o label mancanti


            # Salva i dati processati
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.token_type_ids.append(token_type_ids)
            self.intent_labels.append(intent_label_id)
            self.slot_labels.append(torch.LongTensor(aligned_slot_labels)) # Salva come LongTensor


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Restituisce i dati tokenizzati e allineati come tensori
        sample = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'token_type_ids': self.token_type_ids[idx],
            'intent_labels': self.intent_labels[idx], # Già un int o tensor
            'slot_labels': self.slot_labels[idx]      # Già un LongTensor
        }
        return sample

# --- Funzione Collate (per DataLoader con input BERT) ---
def collate_fn_bert(data):
    '''
    Funzione custom per creare un batch da campioni tokenizzati BERT.
    Gestisce lo stacking dei tensori e li sposta sul device.
    Non ha bisogno di fare padding se Dataset lo ha già fatto con 'max_length'.
    '''
    # Data è una lista di dizionari, ciascuno con 'input_ids', 'attention_mask', etc.
    # Stack i tensori per ogni chiave
    input_ids = torch.stack([d['input_ids'] for d in data])
    attention_masks = torch.stack([d['attention_mask'] for d in data])
    token_type_ids = torch.stack([d['token_type_ids'] for d in data])

    # Intent labels sono int, raccoglili e poi stack
    # Assicurati che siano LongTensor per CrossEntropyLoss
    intent_labels = torch.LongTensor([d['intent_labels'] for d in data])

    # Slot labels sono già LongTensor, stackli
    slot_labels = torch.stack([d['slot_labels'] for d in data])


    # Sposta tutti i tensori sul device designato
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
    }
    return batch

# --- Inizializzazione Pesi (Opzionale per i nuovi layer) ---
# useremo questa funzione solo per i layer lineari aggiunti sopra BERT
def init_weights_linear(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        # Non reinizializzare i pesi di BERT!

# --- END OF FILE utils.py ---