
import json
import os
from collections import Counter
import torch
import torch.nn as nn # For type hinting if init_weights_linear is kept
import torch.utils.data as data
from sklearn.model_selection import train_test_split # Imported for use by main.py
from transformers import BertTokenizerFast

# ==============================================================================
# --- Global Variables and Device Configuration ---
# ==============================================================================

if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda:0")
else: device = torch.device("cpu")

if device.type == "cuda": os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PAD_TOKEN_ID = 0 # Default, can be overridden by tokenizer
IGNORE_INDEX = -100 # For CrossEntropyLoss with BERT slots

# ==============================================================================
# --- Data Loading Function ---
# ==============================================================================

def load_data(path):
    """Loads a dataset from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f: return json.loads(f.read())
    except FileNotFoundError: print(f"CRITICAL ERROR: Data file not found at {path}."); exit()
    except json.JSONDecodeError: print(f"CRITICAL ERROR: Could not decode JSON from {path}."); exit()

# ==============================================================================
# --- Lang Class (Label Vocabulary Management for BERT) ---
# ==============================================================================

class Lang:
    """Manages mappings for intent/slot labels to numerical IDs for BERT."""
    def __init__(self, intent_labels_set, slot_labels_set):
        self.slot2id = self._build_label_vocab(slot_labels_set, add_pad=True, pad_id_val=PAD_TOKEN_ID)
        self.intent2id = self._build_label_vocab(intent_labels_set, add_pad=False)
        self.id2slot = {idx: slot for slot, idx in self.slot2id.items()}
        self.id2intent = {idx: intent for intent, idx in self.intent2id.items()}

    def _build_label_vocab(self, labels_set, add_pad=True, pad_id_val=0):
        vocab = {"pad": pad_id_val} if add_pad else {}
        for label in sorted(list(labels_set)):
            if label not in vocab: vocab[label] = len(vocab)
        return vocab

# ==============================================================================
# --- PyTorch Dataset Class (IntentsAndSlotsBERT) ---
# Handles ATIS data tokenization with BERT and aligns slot labels.
# ==============================================================================

class IntentsAndSlotsBERT(data.Dataset):
    """Custom Dataset for ATIS with BERT tokenization and slot label alignment."""
    def __init__(self, raw_dataset_items, lang_obj, tokenizer: BertTokenizerFast, max_len=50):
        self.lang = lang_obj; self.tokenizer = tokenizer; self.max_len = max_len
        self.ignore_index_val = IGNORE_INDEX
        self.processed_input_ids, self.processed_attention_masks, self.processed_token_type_ids = [], [], []
        self.processed_intent_labels, self.processed_slot_labels = [], []
        self.original_words_list_batch, self.original_slot_labels_list_batch, self.word_ids_list_batch = [], [], []

        for item in raw_dataset_items:
            utterance_text = item["utterance"]
            original_words = utterance_text.split(); original_slot_labels_str = item["slots"].split()
            encoded_utterance = tokenizer.encode_plus(
                utterance_text, max_length=self.max_len, padding="max_length", truncation=True,
                return_attention_mask=True, return_token_type_ids=True, return_tensors="pt"
            )
            current_input_ids = encoded_utterance["input_ids"].squeeze(0)
            current_word_ids = encoded_utterance.word_ids(batch_index=0)
            if current_word_ids is None: print(f"Warning: word_ids not obtained for: {utterance_text}. Skipping."); continue

            aligned_slot_label_ids = [self.ignore_index_val] * len(current_input_ids)
            previous_word_idx = -1
            for token_pos, word_idx in enumerate(current_word_ids):
                if word_idx is not None: # Token corresponds to a word
                    if word_idx != previous_word_idx: # First sub-token of the word
                        if word_idx < len(original_slot_labels_str):
                            aligned_slot_label_ids[token_pos] = self.lang.slot2id.get(original_slot_labels_str[word_idx], self.ignore_index_val)
                    previous_word_idx = word_idx
            
            self.processed_input_ids.append(current_input_ids)
            self.processed_attention_masks.append(encoded_utterance["attention_mask"].squeeze(0))
            self.processed_token_type_ids.append(encoded_utterance["token_type_ids"].squeeze(0))
            self.processed_intent_labels.append(self.lang.intent2id.get(item["intent"], -1))
            self.processed_slot_labels.append(torch.LongTensor(aligned_slot_label_ids))
            self.original_words_list_batch.append(original_words)
            self.original_slot_labels_list_batch.append(original_slot_labels_str)
            self.word_ids_list_batch.append(current_word_ids)

    def __len__(self): return len(self.processed_input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.processed_input_ids[idx], "attention_mask": self.processed_attention_masks[idx],
            "token_type_ids": self.processed_token_type_ids[idx], "intent_labels": self.processed_intent_labels[idx],
            "slot_labels": self.processed_slot_labels[idx], "original_words": self.original_words_list_batch[idx],
            "original_slot_labels": self.original_slot_labels_list_batch[idx], "word_ids": self.word_ids_list_batch[idx],
        }

# ==============================================================================
# --- Collate Function for DataLoader (BERT) ---
# ==============================================================================

def collate_fn_bert(batch_samples):
    """Collates BERT samples: stacks tensors, moves to device, keeps lists as Python lists."""
    input_ids = torch.stack([s["input_ids"] for s in batch_samples])
    attention_masks = torch.stack([s["attention_mask"] for s in batch_samples])
    token_type_ids = torch.stack([s["token_type_ids"] for s in batch_samples])
    intent_labels = torch.LongTensor([s["intent_labels"] for s in batch_samples])
    slot_labels = torch.stack([s["slot_labels"] for s in batch_samples])

    return { # Tensors moved to device, lists remain as Python lists
        "input_ids": input_ids.to(device), "attention_mask": attention_masks.to(device),
        "token_type_ids": token_type_ids.to(device), "intent_labels": intent_labels.to(device),
        "slot_labels": slot_labels.to(device),
        "original_words": [s["original_words"] for s in batch_samples],
        "original_slot_labels": [s["original_slot_labels"] for s in batch_samples],
        "word_ids": [s["word_ids"] for s in batch_samples],
    }

# ==============================================================================
# --- Weight Initialization (Optional for new Linear Layers) ---
# ==============================================================================

def init_weights_linear(module_with_linear_layers):
    """Initializes weights for nn.Linear layers (if not using default init)."""
    for m in module_with_linear_layers.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None: m.bias.data.fill_(0.01)
