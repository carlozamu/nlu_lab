import torch
import torch.utils.data as data  

def read_file(path: str, eos_token: str = "<eos>") -> list[str]:
    """
    Read text file and add end-of-sentence token to each line.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def get_vocab(corpus: list[str], special_tokens: list[str] = []) -> dict[str, int]:
    """
    Build vocabulary mapping from words to integer IDs.
    Special tokens are added first with lower IDs.
    """
    output = {}
    i = 0
    # Add special tokens first (e.g., <pad>, <eos>)
    for st in special_tokens:
        output[st] = i
        i += 1
    # Add all unique words from corpus
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class Lang:
    """
    Language class that manages vocabulary mappings.
    Creates word2id and id2word dictionaries for the corpus.
    """
    def __init__(self, corpus: list[str], special_tokens: list[str] = []):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    def get_vocab(self, corpus, special_tokens=[]):
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
    PyTorch Dataset for Penn TreeBank language modeling.
    Creates source-target pairs where target is shifted by one position.
    """
    def __init__(self, corpus: list[str], lang: Lang):
        self.source = []
        self.target = []
        
        # Create source-target pairs for language modeling
        # Source: [w1, w2, w3, ..., wn]
        # Target: [w2, w3, w4, ..., wn+1]
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])  # All words except last
            self.target.append(sentence.split()[1:])    # All words except first
        
        # Convert words to integer IDs
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang):
        """
        Convert word sequences to ID sequences using vocabulary.
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
    Collate function for DataLoader: handles variable-length sequences.
    Pads sequences to same length and moves to GPU/CPU.
    """
    def merge(sequences):
        """
        Pad sequences to the same length using pad_token_id.
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token_id)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  
        padded_seqs = padded_seqs.detach()  
        return padded_seqs, lengths
    
    # Sort batch by sequence length (longest first) for efficient processing
    batch_data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in batch_data[0].keys():
        new_item[key] = [d[key] for d in batch_data]

    # Pad sequences and move to device
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)  # Total tokens in batch for loss normalization
    return new_item