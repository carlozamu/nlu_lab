import json
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import BertTokenizerFast

# ========== DEVICE CONFIGURATION ==========
# Automatically select the best available device for training
if torch.backends.mps.is_available():
    device = torch.device("mps")          # Apple Silicon GPU
elif torch.cuda.is_available():
    device = torch.device("cuda:0")       # NVIDIA GPU
else:
    device = torch.device("cpu")          # CPU fallback

if device.type == "cuda":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better CUDA error reporting

# ========== CONSTANTS ==========
PAD_TOKEN = 0        # ID for padding tokens
IGNORE_INDEX = -100  # Special index to ignore in loss calculation (for padding)


def load_data(path):
    """Load JSON dataset from file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Data file not found at {path}.")
        exit()
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Could not decode JSON from {path}.")
        exit()


class Lang:
    """
    Language class that creates mappings between labels and IDs.
    
    This is essential for converting text labels (like "book_flight", "B-city") 
    into numerical IDs that the model can work with.
    """

    def __init__(self, intent_labels_set, slot_labels_set):
        # Create label-to-ID mappings
        self.slot2id = self._build_label_vocab(
            slot_labels_set, add_pad=True, pad_id_val=PAD_TOKEN
        )
        self.intent2id = self._build_label_vocab(intent_labels_set, add_pad=False)
        
        # Create reverse mappings (ID-to-label) for decoding predictions
        self.id2slot = {idx: slot for slot, idx in self.slot2id.items()}
        self.id2intent = {idx: intent for intent, idx in self.intent2id.items()}

    def _build_label_vocab(self, labels_set, add_pad=True, pad_id_val=0):
        """Build vocabulary mapping from labels to IDs."""
        vocab = {"pad": pad_id_val} if add_pad else {}
        for label in sorted(list(labels_set)):
            if label not in vocab:
                vocab[label] = len(vocab)
        return vocab


class IntentsAndSlotsBERT(data.Dataset):
    """
    Dataset class for BERT-based joint intent and slot classification.
    
    This class handles the complex task of:
    1. Tokenizing text with BERT tokenizer (which creates subwords)
    2. Aligning slot labels with BERT tokens (challenging because one word can become multiple tokens)
    3. Preparing data in the format expected by BERT
    """

    def __init__(
        self, raw_dataset_items, lang_obj, tokenizer: BertTokenizerFast, max_len=50
    ):
        self.lang = lang_obj
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignore_index_val = IGNORE_INDEX
        
        # Storage for processed data
        (
            self.processed_input_ids,           # BERT token IDs
            self.processed_attention_masks,     # Attention masks for padding
            self.processed_token_type_ids,      # Segment IDs for BERT
        ) = [], [], []
        self.processed_intent_labels, self.processed_slot_labels = [], []
        
        # Keep original data for evaluation alignment
        (
            self.original_words_list_batch,
            self.original_slot_labels_list_batch,
            self.word_ids_list_batch,           # Maps BERT tokens back to original words
        ) = [], [], []

        # Process each training example
        for item in raw_dataset_items:
            utterance_text = item["utterance"]
            original_words = utterance_text.split()
            original_slot_labels_str = item["slots"].split()
            
            # BERT tokenization with special tokens and padding
            encoded_utterance = tokenizer.encode_plus(
                utterance_text,
                max_length=self.max_len,
                padding="max_length",           # Pad to max length
                truncation=True,                # Truncate if too long
                return_attention_mask=True,     # Return attention mask
                return_token_type_ids=True,     # Return segment IDs
                return_tensors="pt",            # Return PyTorch tensors
            )
            
            current_input_ids = encoded_utterance["input_ids"].squeeze(0)
            current_word_ids = encoded_utterance.word_ids(batch_index=0)  # Critical for alignment!
            
            if current_word_ids is None:
                print(
                    f"Warning: word_ids not obtained for: {utterance_text}. Skipping."
                )
                continue

            # SLOT LABEL ALIGNMENT - This is the tricky part!
            # BERT splits words into subwords, but we have labels for whole words
            # We need to align subword tokens back to original word labels
            aligned_slot_label_ids = [self.ignore_index_val] * len(current_input_ids)
            previous_word_idx = -1
            
            for token_pos, word_idx in enumerate(current_word_ids):
                if word_idx is not None:  # Token corresponds to a word (not special token)
                    if word_idx != previous_word_idx:  # First sub-token of the word
                        if word_idx < len(original_slot_labels_str):
                            # Assign the word's label to the first subtoken only
                            aligned_slot_label_ids[token_pos] = self.lang.slot2id.get(
                                original_slot_labels_str[word_idx],
                                self.ignore_index_val,
                            )
                    # Other subtokens of the same word get IGNORE_INDEX
                    previous_word_idx = word_idx

            # Store processed data
            self.processed_input_ids.append(current_input_ids)
            self.processed_attention_masks.append(
                encoded_utterance["attention_mask"].squeeze(0)
            )
            self.processed_token_type_ids.append(
                encoded_utterance["token_type_ids"].squeeze(0)
            )
            self.processed_intent_labels.append(
                self.lang.intent2id.get(item["intent"], -1)
            )
            self.processed_slot_labels.append(torch.LongTensor(aligned_slot_label_ids))
            
            # Keep original data for evaluation
            self.original_words_list_batch.append(original_words)
            self.original_slot_labels_list_batch.append(original_slot_labels_str)
            self.word_ids_list_batch.append(current_word_ids)

    def __len__(self):
        return len(self.processed_input_ids)

    def __getitem__(self, idx):
        """Return a single training example with all necessary data."""
        return {
            "input_ids": self.processed_input_ids[idx],
            "attention_mask": self.processed_attention_masks[idx],
            "token_type_ids": self.processed_token_type_ids[idx],
            "intent_labels": self.processed_intent_labels[idx],
            "slot_labels": self.processed_slot_labels[idx],
            "original_words": self.original_words_list_batch[idx],
            "original_slot_labels": self.original_slot_labels_list_batch[idx],
            "word_ids": self.word_ids_list_batch[idx],
        }


def collate_fn_bert(batch_samples):
    """
    Collate function to batch multiple samples together.
    
    This function:
    1. Stacks tensors for efficient batch processing
    2. Moves tensors to the appropriate device (GPU/CPU)
    3. Keeps non-tensor data (like original words) as Python lists
    """
    # Stack tensors into batches
    input_ids = torch.stack([s["input_ids"] for s in batch_samples])
    attention_masks = torch.stack([s["attention_mask"] for s in batch_samples])
    token_type_ids = torch.stack([s["token_type_ids"] for s in batch_samples])
    intent_labels = torch.LongTensor([s["intent_labels"] for s in batch_samples])
    slot_labels = torch.stack([s["slot_labels"] for s in batch_samples])

    return {
        # Move tensors to device for model computation
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device),
        "token_type_ids": token_type_ids.to(device),
        "intent_labels": intent_labels.to(device),
        "slot_labels": slot_labels.to(device),
        # Keep original data as lists for evaluation
        "original_words": [s["original_words"] for s in batch_samples],
        "original_slot_labels": [s["original_slot_labels"] for s in batch_samples],
        "word_ids": [s["word_ids"] for s in batch_samples],
    }


def init_weights_linear(module_with_linear_layers):
    """Initialize linear layer weights with small uniform values."""
    for m in module_with_linear_layers.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)