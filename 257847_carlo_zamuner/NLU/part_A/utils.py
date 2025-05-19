
import json
import os
# import random # Not used directly in this version
from collections import Counter
# from pprint import pprint # Not used directly in this version

import numpy as np
import torch
import torch.nn as nn # Used in init_weights
import torch.utils.data as data
from sklearn.model_selection import train_test_split # Imported but seems unused in this file directly,
                                                     # might be used by main.py that imports * from utils

# ==============================================================================
# --- Global Variables and Device Configuration ---
# ==============================================================================

# Determine the device to use: MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0") # Use the first available GPU
else:
    device = torch.device("cpu")

# print(f"Device selected for PyTorch: {device}") # Optional: Keep for debugging startup

if device.type == "cuda":
    # For detailed CUDA error reporting (useful for debugging)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Index for the padding token, used across dataset and model
PAD_TOKEN = 0

# ==============================================================================
# --- Data Loading Function ---
# ==============================================================================

def load_data(path):
    """
    Loads a dataset from a JSON file.
    Args:
        path (str): Path to the JSON data file.
    Returns:
        list: A list of dictionaries representing the dataset.
    """
    dataset = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
    except FileNotFoundError:
        # Critical error if data file is not found
        print(f"CRITICAL ERROR: Data file not found at {path}")
        print("Ensure ATIS dataset files (train.json, test.json) are correctly placed.")
        exit()
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Could not decode JSON from {path}. Check file format.")
        exit()
    return dataset

# ==============================================================================
# --- Lang Class (Vocabulary Management for Words, Intents, Slots) ---
# ==============================================================================

class Lang:
    """
    Helper class to manage mappings between words/labels and numerical indices.
    Used for LSTM model's vocabulary and label encoding.
    """
    def __init__(self, words_for_vocab, intent_labels, slot_labels, cutoff_freq=0):
        """
        Initializes word2id, slot2id, intent2id dictionaries and their inverses.
        Args:
            words_for_vocab (list): List of all words from the training set (for vocabulary).
            intent_labels (set): Set of all unique intent labels.
            slot_labels (set): Set of all unique slot labels.
            cutoff_freq (int): Minimum frequency for a word to be included in the vocabulary.
        """
        self.word2id = self._build_word_vocab(words_for_vocab, cutoff_freq, add_unk=True)
        self.slot2id = self._build_label_vocab(slot_labels, add_pad=True)
        self.intent2id = self._build_label_vocab(intent_labels, add_pad=False) # Intents typically don't need padding

        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.id2slot = {idx: slot for slot, idx in self.slot2id.items()}
        self.id2intent = {idx: intent for intent, idx in self.intent2id.items()}

    def _build_word_vocab(self, word_list, cutoff_freq, add_unk=True):
        """Builds the word -> id dictionary."""
        vocab = {"pad": PAD_TOKEN}
        if add_unk:
            vocab["unk"] = len(vocab) # 'unk' token for out-of-vocabulary words
        
        word_counts = Counter(word_list)
        for word, count in word_counts.items():
            if count > cutoff_freq:
                vocab[word] = len(vocab)
        return vocab

    def _build_label_vocab(self, labels, add_pad=True):
        """Builds the label -> id dictionary (for slots and intents)."""
        vocab = {}
        if add_pad: # Padding is typically needed for slot sequences
            vocab["pad"] = PAD_TOKEN
        
        # Sort elements to ensure consistent mapping across runs
        for label in sorted(list(labels)):
            vocab[label] = len(vocab)
        return vocab

# ==============================================================================
# --- PyTorch Dataset Class (IntentsAndSlots for LSTM) ---
# ==============================================================================

class IntentsAndSlots(data.Dataset):
    """
    Custom PyTorch Dataset for handling the ATIS dataset for the LSTM model.
    Converts text and labels into sequences of numerical indices.
    """
    def __init__(self, raw_dataset_items, lang_obj, unk_token_str="unk"):
        """
        Args:
            raw_dataset_items (list): List of raw data dictionaries (utterance, slots, intent).
            lang_obj (Lang): Instantiated Lang object containing vocabularies.
            unk_token_str (str): String representation of the unknown token.
        """
        self.utterances_str = []
        self.intents_str = []
        self.slots_str_sequences = []
        self.unk_token = unk_token_str

        for item in raw_dataset_items:
            self.utterances_str.append(item["utterance"])
            self.slots_str_sequences.append(item["slots"]) # Slots are space-separated strings
            self.intents_str.append(item["intent"])

        # Convert text sequences to ID sequences using the Lang object
        self.utterance_ids_list = self._map_sequences_to_ids(self.utterances_str, lang_obj.word2id)
        self.slot_ids_list = self._map_sequences_to_ids(self.slots_str_sequences, lang_obj.slot2id)
        self.intent_ids_list = self._map_labels_to_ids(self.intents_str, lang_obj.intent2id)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.utterances_str)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (utterance, slots, intent) as Tensors.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary containing 'utterance' (LongTensor), 'slots' (LongTensor),
                  and 'intent' (int ID).
        """
        # Convert ID lists to LongTensors for PyTorch model consumption
        utterance_tensor = torch.LongTensor(self.utterance_ids_list[idx])
        slots_tensor = torch.LongTensor(self.slot_ids_list[idx])
        intent_id = self.intent_ids_list[idx] # Intent is a single ID

        return {"utterance": utterance_tensor, "slots": slots_tensor, "intent": intent_id}

    def _map_labels_to_ids(self, label_data, mapper_dict):
        """Maps a list of labels (e.g., intents) to their numerical IDs."""
        # Fallback to 'unk' ID if label not in mapper, then to PAD_TOKEN if 'unk' also not in mapper
        # (though for intents/slots, all labels should ideally be in the mapper).
        unk_id = mapper_dict.get(self.unk_token, PAD_TOKEN)
        return [mapper_dict.get(label, unk_id) for label in label_data]

    def _map_sequences_to_ids(self, text_sequences, mapper_dict):
        """Maps a list of text sequences (utterances, slots) to lists of numerical IDs."""
        id_sequences = []
        unk_id = mapper_dict.get(self.unk_token, PAD_TOKEN) # Fallback for unknown tokens
        for seq_str in text_sequences:
            token_ids = [mapper_dict.get(token, unk_id) for token in str(seq_str).split()]
            id_sequences.append(token_ids)
        return id_sequences

# ==============================================================================
# --- Collate Function for DataLoader (LSTM) ---
# ==============================================================================

def collate_fn(batch_samples):
    """
    Custom collate function for DataLoader.
    Processes a list of samples (from __getitem__) to form a batch.
    It sorts sequences by length (for potential packing), pads them,
    and moves tensors to the designated device.
    Args:
        batch_samples (list): A list of dictionaries, where each dict is an output of __getitem__.
    Returns:
        dict: A batch dictionary ready for model input.
    """
    # Sort batch by utterance length in descending order (often useful for RNNs with packing)
    batch_samples.sort(key=lambda x: len(x["utterance"]), reverse=True)

    # Group data by keys ('utterance', 'slots', 'intent')
    batched_data = {}
    for key in batch_samples[0].keys():
        batched_data[key] = [sample[key] for sample in batch_samples]

    # Pad utterance and slot sequences
    utterance_seqs_padded, utterance_lengths = _pad_sequences(batched_data["utterance"])
    slot_seqs_padded, _ = _pad_sequences(batched_data["slots"]) # Slot lengths match utterance_lengths

    intent_ids_tensor = torch.LongTensor(batched_data["intent"])

    # Move tensors to the configured device
    utterance_seqs_padded = utterance_seqs_padded.to(device)
    slot_seqs_padded = slot_seqs_padded.to(device)
    intent_ids_tensor = intent_ids_tensor.to(device)
    utterance_lengths = utterance_lengths.to(device) # Keep on device as per original for consistency

    return {
        "utterances": utterance_seqs_padded, # Padded input sequences
        "intents": intent_ids_tensor,        # Intent labels
        "y_slots": slot_seqs_padded,         # Padded ground truth slot labels
        "slots_len": utterance_lengths,      # Original sequence lengths (used for packing, loss, metrics)
    }

def _pad_sequences(sequences_list):
    """
    Helper function to pad a list of sequences (tensor list) to the max length in the list.
    Args:
        sequences_list (list): List of 1D Tensors (sequences of IDs).
    Returns:
        tuple: (padded_sequences_tensor, lengths_tensor)
    """
    lengths = [len(seq) for seq in sequences_list]
    max_len = max(lengths) if lengths else 0
    
    padded_sequences_tensor = torch.LongTensor(len(sequences_list), max_len).fill_(PAD_TOKEN)
    for i, seq_tensor in enumerate(sequences_list):
        end_idx = lengths[i]
        padded_sequences_tensor[i, :end_idx] = seq_tensor[:end_idx]
        
    return padded_sequences_tensor, torch.LongTensor(lengths)

# ==============================================================================
# --- Weight Initialization Function ---
# ==============================================================================

def init_weights(model_module):
    """
    Initializes weights of RNN and Linear layers in the model.
    Args:
        model_module (nn.Module): The model to initialize.
    """
    for module in model_module.modules():
        if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in module.named_parameters():
                if "weight_ih" in name: # Input-to-hidden weights
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name: # Hidden-to-hidden weights
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0) # Biases to zero
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01) # Small uniform for linear weights
            if module.bias is not None:
                module.bias.data.fill_(0.01) # Small constant for linear biases

