import torch
import torch.utils.data as data  # Used as a base class for PennTreeBank


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
    with open(path, "r", encoding="utf-8") as f:  # Added encoding for robustness
        for line in f:  # Iterate directly over file object
            stripped_line = line.strip()
            if stripped_line:  # Process only non-empty lines
                output.append(stripped_line + " " + eos_token)
    return output


# ==============================================================================
# --- VOCABULARY CREATION FUNCTION ---
# ==============================================================================
def get_vocab(corpus: list[str], special_tokens: list[str] = []) -> dict[str, int]:
    """
    Creates a word-to-id dictionary from a corpus and special tokens.
    Args:
        corpus (list[str]): List of sentences (strings).
        special_tokens (list[str]): List of special tokens to add first.
    Returns:
        dict[str, int]: Vocabulary mapping words to unique integer IDs.
    """
    vocab = {}
    idx = 0
    # Add special tokens first to ensure consistent IDs
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

    def __init__(self, corpus: list[str], special_tokens: list[str] = []):
        """
        Args:
            corpus (list[str]): List of sentences to build vocabulary from.
            special_tokens (list[str]): Special tokens like <pad>, <eos>.
        """
        # Use the external get_vocab function
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    # Removed duplicated get_vocab method from the original class


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
            corpus (list[str]): List of sentences (strings with EOS token).
            lang_obj (Lang): Instance of the Lang class containing the vocabulary.
        """
        sources_as_tokens = []
        targets_as_tokens = []
        for sentence in corpus:
            tokens = sentence.split()
            if (
                len(tokens) > 1
            ):  # Ensure there's at least one token for source and one for target
                sources_as_tokens.append(
                    tokens[:-1]
                )  # All but the last token (e.g., <eos>)
                targets_as_tokens.append(tokens[1:])  # All but the first token
        # Convert token sequences to ID sequences
        self.source_ids = self._map_sequences_to_ids(
            sources_as_tokens, lang_obj.word2id
        )
        self.target_ids = self._map_sequences_to_ids(
            targets_as_tokens, lang_obj.word2id
        )

        # Ensure source and target lists have the same length after OOV filtering
        min_len = min(len(self.source_ids), len(self.target_ids))
        self.source_ids = self.source_ids[:min_len]
        self.target_ids = self.target_ids[:min_len]

    def _map_sequences_to_ids(
        self, sequences_of_tokens: list[list[str]], word2id_dict: dict[str, int]
    ) -> list[list[int]]:
        """
        Helper method to convert lists of token sequences to lists of ID sequences.
        Handles Out-Of-Vocabulary (OOV) words.
        """
        id_sequences = []
        unk_token_id = word2id_dict.get("<unk>")  # Get ID for <unk> token, if it exists

        for token_sequence in sequences_of_tokens:
            current_ids = []
            valid_sequence = True
            for token in token_sequence:
                token_id = word2id_dict.get(token, unk_token_id)
                if token_id is None:  # OOV and no <unk> token defined
                    print(
                        f"Warning: OOV token '{token}' found and no '<unk>' token in vocabulary. Skipping sequence: '{' '.join(token_sequence)}'"
                    )
                    valid_sequence = False
                    break
                current_ids.append(token_id)
            if valid_sequence:
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
def collate_fn(batch_data, pad_token_id, device):
    """
    Processes a list of samples to form a batch.
    Sorts by source length, pads sequences, moves to device, and calculates total non-pad tokens.

    Args:
        batch_samples (list[dict]): A list of samples, where each sample is a dict
                                   {'source': Tensor, 'target': Tensor}.
        pad_token_id (int): The ID of the padding token.
        device (torch.device): The device to move tensors to.

    Returns:
        dict: A batch dictionary with 'source', 'target' (padded tensors on device),
              and 'number_tokens' (total non-pad tokens in the batch).
    """
    # Sort batch by source sequence length in descending order (common for RNNs)
    batch_data.sort(key=lambda x: len(x["source"]), reverse=True)

    # Separate source and target tensors from the batch
    source_tensors = [sample["source"] for sample in batch_data]
    target_tensors = [sample["target"] for sample in batch_data]

    # Helper function to pad a list of sequences (1D tensors)
    def _pad_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0  # Handle empty sequences list

        # Create a tensor filled with pad_token_id
        padded_tensor = torch.LongTensor(len(sequences), max_len).fill_(pad_token_id)

        # Copy each sequence into the padded tensor
        for i, seq in enumerate(sequences):
            end_idx = lengths[i]
            padded_tensor[i, :end_idx] = seq[:end_idx]  # Ensure correct slicing
        return padded_tensor, lengths

    # Pad source and target sequences
    padded_sources, source_lengths = _pad_sequences(source_tensors)
    # Target lengths will be the same as source lengths for language modeling
    padded_targets, target_lengths = _pad_sequences(target_tensors)

    # Prepare the collated batch dictionary
    collated_batch = {
        "source": padded_sources.to(device),
        "target": padded_targets.to(device),
        "number_tokens": sum(
            target_lengths
        ),  # Sum of actual (non-padded) token lengths
    }
    return collated_batch
