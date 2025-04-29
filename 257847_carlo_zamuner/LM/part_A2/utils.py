import torch
import torch.utils.data as data


# Funzione originale read_file
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


# Funzione originale get_vocab
def get_vocab(corpus, special_tokens=[]):
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


# Classe originale Lang
class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):  # Metodo duplicato originale
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


# Classe originale PennTreeBank
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source, self.target = [], []
        for sentence in corpus:
            tokens = sentence.split()
            if len(tokens) > 1:  # Evita righe vuote
                self.source.append(tokens[0:-1])
                self.target.append(tokens[1:])
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(
            self.source_ids
        )  # Usa source_ids per lunghezza corretta post-mapping

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {"source": src, "target": trg}

    # Metodo originale mapping_seq
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print("OOV found!")
                    print("Handle OOV!")
                    break
            res.append(tmp_seq)
        # Ritorna lista, potrebbe avere seq vuote se OOV trovato e break usato
        return res


# Funzione originale collate_fn
def collate_fn(data, pad_token, DEVICE):  # Accetta pad_token e DEVICE
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    # Assunzione: data[0] esiste (batch non vuoto)
    if not data:
        return None  # Gestisce batch vuoto
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item
