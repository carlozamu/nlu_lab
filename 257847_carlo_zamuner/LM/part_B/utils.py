import torch
import torch.utils.data as data  


def read_file(path: str, eos_token: str = "<eos>") -> list[str]:
    """
    Reads a text file and processes it for language modeling.
    Each line becomes a sentence with an end-of-sentence token appended.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def get_vocab(corpus: list[str], special_tokens: list[str] = []) -> dict[str, int]:
    """
    Creates a vocabulary mapping from words to integer IDs.
    Special tokens (like <pad>, <eos>) are assigned first for consistent indexing.
    """
    output = {}
    i = 0
    # Assign IDs to special tokens first
    for st in special_tokens:
        output[st] = i
        i += 1
    # Then assign IDs to all unique words in the corpus
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class Lang:
    """
    Language processing class that manages word-to-ID and ID-to-word mappings.
    This is essential for converting text to numerical format for neural networks.
    """

    def __init__(self, corpus: list[str], special_tokens: list[str] = []):
        # Create word-to-ID mapping
        self.word2id = self.get_vocab(corpus, special_tokens)
        # Create reverse mapping (ID-to-word) for decoding
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    def get_vocab(self, corpus, special_tokens=[]):
        """Same functionality as the standalone get_vocab function"""
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


class PennTreeBank(data.Dataset):
    """
    PyTorch Dataset class for Penn Tree Bank language modeling data.
    Converts text sequences into source-target pairs for next-word prediction.
    
    For language modeling, each sequence is split into:
    - source: words[0:-1] (all words except last)
    - target: words[1:] (all words except first)
    This creates the next-word prediction task.
    """

    def __init__(self, corpus: list[str], lang: Lang):
        self.source = []
        self.target = []
        
        # Create source-target pairs for language modeling
        for sentence in corpus:
            words = sentence.split()
            self.source.append(words[0:-1])  # Input sequence (exclude last word)
            self.target.append(words[1:])    # Target sequence (exclude first word)
        
        # Convert words to integer IDs using the language mapping
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        """Return a single sample as tensors"""
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang):
        """
        Convert word sequences to ID sequences using the language vocabulary.
        Handles out-of-vocabulary (OOV) words by stopping processing.
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') 
                    break
            res.append(tmp_seq)
        return res


def collate_fn(batch_data, pad_token_id, device):
    """
    Custom collate function for DataLoader that handles variable-length sequences.
    This function:
    1. Sorts sequences by length (longest first) for efficient batching
    2. Pads shorter sequences to match the longest in the batch
    3. Moves data to the specified device (CPU/GPU)
    4. Calculates total number of tokens for loss averaging
    """
    def merge(sequences):
        """Pad sequences to the same length"""
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        
        # Create padded tensor filled with pad_token_id
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token_id)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # Copy actual sequence data
        padded_seqs = padded_seqs.detach()  
        return padded_seqs, lengths
    
    # Sort batch by sequence length (descending) for efficient processing
    batch_data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in batch_data[0].keys():
        new_item[key] = [d[key] for d in batch_data]

    # Pad sequences to uniform length
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    # Move to device and add token count for loss computation
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)  # Total tokens for proper loss averaging
    return new_item