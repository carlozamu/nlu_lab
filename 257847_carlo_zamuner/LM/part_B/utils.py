
import torch
import torch.utils.data as data # Used as a base class for PennTreeBank

# ==============================================================================
# --- FILE READING FUNCTION ---
# ==============================================================================
def read_file(path: str, eos_token: str = "<eos>") -> list[str]:
    """
    Reads lines from a file, strips whitespace, appends an EOS token,
    and returns a list of processed sentences.

    Args:
        path (str): Path to the text file.
        eos_token (str): End-of-sentence token to append.
    Returns:
        list[str]: List of sentences, each ending with eos_token.
    """
    output = []
    try:
        with open(path, "r", encoding="utf-8") as f: # Added encoding for robustness
            for line in f:
                stripped_line = line.strip()
                if stripped_line: # Process only non-empty lines
                    output.append(stripped_line + " " + eos_token)
    except FileNotFoundError:
        print(f"Error: File not found at path {path}")
        # Depending on desired behavior, could raise error or return empty list
        # raise
    return output

# ==============================================================================
# --- VOCABULARY CREATION FUNCTION ---
# ==============================================================================
def get_vocab(corpus: list[str], special_tokens: list[str] = None) -> dict[str, int]:
    """
    Creates a word-to-id dictionary from a corpus and optional special tokens.

    Args:
        corpus (list[str]): List of sentences (strings).
        special_tokens (list[str], optional): List of special tokens to add first.
                                              Defaults to None, then initialized to empty list.
    Returns:
        dict[str, int]: Vocabulary mapping words to unique integer IDs.
    """
    if special_tokens is None:
        special_tokens = [] # Ensure special_tokens is always a list

    vocab = {}
    idx = 0
    # Add special tokens first
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1
    # Add words from the corpus
    for sentence in corpus:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# ==============================================================================
# --- LANGUAGE UTILITY CLASS ---
# ==============================================================================
class Lang:
    """
    Manages vocabulary, providing word-to-id and id-to-word mappings.
    """
    def __init__(self, corpus: list[str], special_tokens: list[str] = None):
        """
        Args:
            corpus (list[str]): List of sentences to build vocabulary from.
            special_tokens (list[str], optional): Special tokens like <pad>, <eos>.
                                                  Defaults to None, then initialized to empty list.
        """
        if special_tokens is None:
            special_tokens = []
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

# ==============================================================================
# --- PennTreeBank DATASET CLASS ---
# ==============================================================================
class PennTreeBank(data.Dataset):
    """
    PyTorch Dataset for the Penn Treebank, creating source/target pairs for
    language modeling.
    """
    def __init__(self, corpus: list[str], lang_obj: Lang):
        """
        Args:
            corpus (list[str]): List of sentences (strings, expected to end with EOS).
            lang_obj (Lang): Instance of the Lang class containing the vocabulary.
        """
        sources_as_tokens: list[list[str]] = []
        targets_as_tokens: list[list[str]] = []

        for sentence in corpus:
            tokens = sentence.split()
            # Ensure sentence has enough tokens for a source and a target
            if len(tokens) > 1:
                sources_as_tokens.append(tokens[:-1]) # All tokens except the last one
                targets_as_tokens.append(tokens[1:])  # All tokens except the first one

        # Convert token sequences to ID sequences
        self.source_ids = self._map_sequences_to_ids(sources_as_tokens, lang_obj.word2id)
        self.target_ids = self._map_sequences_to_ids(targets_as_tokens, lang_obj.word2id)

        # Critical: Ensure source and target lists have the same length after OOV filtering
        # This can happen if a sequence is skipped in _map_sequences_to_ids for one list but not the other,
        # or if sequences become empty.
        min_len = min(len(self.source_ids), len(self.target_ids))
        self.source_ids = self.source_ids[:min_len]
        self.target_ids = self.target_ids[:min_len]
        
        # Further filter out any pairs where one of the sequences might have become empty
        # after OOV handling, if an OOV causes an entire sequence to be dropped.
        valid_indices = [i for i, (src, tgt) in enumerate(zip(self.source_ids, self.target_ids)) if src and tgt]
        self.source_ids = [self.source_ids[i] for i in valid_indices]
        self.target_ids = [self.target_ids[i] for i in valid_indices]


    def _map_sequences_to_ids(self,
                              sequences_of_tokens: list[list[str]],
                              word2id_dict: dict[str, int]) -> list[list[int]]:
        """
        Helper method to convert lists of token sequences to lists of ID sequences.
        Handles Out-Of-Vocabulary (OOV) words by using <unk> token if available.
        """
        id_sequences = []
        unk_token_id = word2id_dict.get("<unk>") # Get ID for <unk> token

        for token_sequence in sequences_of_tokens:
            current_ids = []
            valid_sequence = True
            for token in token_sequence:
                token_id = word2id_dict.get(token) # Try to get ID
                if token_id is None: # Token not in vocab
                    if unk_token_id is not None: # Use <unk> if available
                        token_id = unk_token_id
                    else: # OOV and no <unk> token defined
                        print(f"Warning: OOV token '{token}' found and no '<unk>' token in vocabulary. "
                              f"Skipping sequence: '{' '.join(token_sequence)}'")
                        valid_sequence = False
                        break # Skip this entire sequence
                current_ids.append(token_id)

            if valid_sequence and current_ids: # Add if valid and not empty
                id_sequences.append(current_ids)
        return id_sequences

    def __len__(self) -> int:
        # The length is the number of valid source/target ID sequence pairs
        return len(self.source_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Returns a single sample (source and target ID tensors)
        source_tensor = torch.LongTensor(self.source_ids[idx])
        target_tensor = torch.LongTensor(self.target_ids[idx])
        return {"source": source_tensor, "target": target_tensor}

# ==============================================================================
# --- COLLATE FUNCTION for DataLoader ---
# ==============================================================================
def collate_fn(batch_samples: list[dict[str, torch.Tensor]],
               pad_token_id: int,
               device: torch.device) -> dict[str, torch.Tensor | int]:
    """
    Processes a list of samples to form a batch.
    Sorts by source length, pads sequences, moves to device, and calculates total non-pad tokens.
    """
    # Filter out any potentially problematic samples (e.g., empty source/target from OOV handling)
    # This step might be redundant if _map_sequences_to_ids already filters thoroughly
    valid_batch_samples = [s for s in batch_samples if len(s["source"]) > 0 and len(s["target"]) > 0]
    if not valid_batch_samples:
        # Return an empty batch or handle as an error if all samples were invalid
        return {"source": torch.LongTensor([]).to(device),
                "target": torch.LongTensor([]).to(device),
                "number_tokens": 0}


    # Sort batch by source sequence length in descending order
    valid_batch_samples.sort(key=lambda x: len(x["source"]), reverse=True)

    source_tensors = [sample['source'] for sample in valid_batch_samples]
    target_tensors = [sample['target'] for sample in valid_batch_samples]

    # Helper function to pad a list of sequences (1D tensors)
    def _pad_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        lengths = [len(seq) for seq in sequences]
        # Handle case where sequences might be empty after filtering
        if not lengths:
            return torch.LongTensor([]), []
        max_len = max(lengths)

        padded_tensor = torch.LongTensor(len(sequences), max_len).fill_(pad_token_id)
        for i, seq in enumerate(sequences):
            end_idx = lengths[i]
            padded_tensor[i, :end_idx] = seq[:end_idx]
        return padded_tensor, lengths

    padded_sources, _ = _pad_sequences(source_tensors) # Source lengths not explicitly needed by LM
    padded_targets, target_lengths = _pad_sequences(target_tensors)

    collated_batch = {
        'source': padded_sources.to(device),
        'target': padded_targets.to(device),
        'number_tokens': sum(target_lengths) # Sum of actual (non-padded) token lengths
    }
    return collated_batch