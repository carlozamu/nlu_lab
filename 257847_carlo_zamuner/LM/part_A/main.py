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
        self.target_ids = self.m