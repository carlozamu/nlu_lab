import json
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split

"""
DATA PROCESSING UTILITIES
-------------------------
This module contains utilities for:
1. Loading and preprocessing ATIS dataset
2. Creating vocabulary mappings for words, slots, and intents
3. Converting text data to numerical indices
4. Batching and padding sequences for efficient training
"""

# Global configuration
device = "cuda:0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
PAD_TOKEN = 0  # Special token ID for padding


def load_data(path):
    """Load dataset from JSON file"""
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang:
    """
    Language processing class that handles vocabulary creation and mapping
    between text and numerical indices for words, slots, and intents
    """
    def __init__(self, words, intents, slots, cutoff=0):
        # Create mappings from text to indices
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)  # No padding for intents
        
        # Create reverse mappings from indices to text
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        """
        Create word-to-index mapping with frequency cutoff and unknown token handling
        """
        vocab = {"pad": PAD_TOKEN}  # Start with padding token
        if unk:
            vocab["unk"] = len(vocab)  # Add unknown token
            
        # Count word frequencies
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:  # Only include words above frequency threshold
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        """
        Create label-to-index mapping for slots or intents
        """
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN  # Slots need padding for variable sequence lengths
            
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    """
    PyTorch Dataset class for Intent and Slot Classification data
    Converts text data to numerical indices and handles the dataset interface
    """
    def __init__(self, dataset, lang, unk="unk"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        # Extract data from dataset
        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        # Convert text to numerical indices using the Lang object
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """Return a single sample as tensors"""
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {"utterance": utt, "slots": slots, "intent": intent}
        return sample

    def mapping_lab(self, data, mapper):
        """Convert single labels (intents) to indices"""
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        """
        Convert sequences (utterances/slots) to indices
        Handles word-by-word conversion for variable-length sequences
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():  # Split by whitespace
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])  # Use unknown token for OOV words
            res.append(tmp_seq)
        return res


def collate_fn(data):
    """
    Custom collate function for DataLoader
    Handles batching of variable-length sequences with padding
    
    This function:
    1. Sorts sequences by length (longest first) for efficient packing
    2. Pads all sequences to the same length within the batch
    3. Moves tensors to GPU
    4. Returns properly formatted batch for the model
    """
    def merge(sequences):
        """Pad sequences to the same length and return lengths"""
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        
        # Create padded tensor filled with PAD_TOKEN
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # Copy actual sequence
            
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort batch by utterance length (descending) for efficient LSTM packing
    data.sort(key=lambda x: len(x["utterance"]), reverse=True)
    
    # Group all samples by key
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Pad utterances and slots to batch max length
    src_utt, _ = merge(new_item["utterance"])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    # Move all tensors to GPU
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    # Return batch in format expected by model
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths  # Actual sequence lengths for packing
    return new_item