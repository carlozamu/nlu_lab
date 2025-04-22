import torch
import torch.utils.data as data
# Rimosso: functools, DataLoader


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
        self.word2id = self.get_vocab(
            corpus, special_tokens
        )  # Chiama il metodo interno
        self.id2word = {v: k for k, v in self.word2id.items()}

    # Metodo originale get_vocab interno alla classe
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


# Classe originale PennTreeBank
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        # Usa metodo originale mapping_seq
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)  # Usa self.source originale

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {"source": src, "target": trg}
        return sample

    # Metodo originale mapping_seq
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    # Gestione OOV originale (solo stampa)
                    print("OOV found!")
                    print("You have to deal with that")
                    # Se si verifica OOV, la sequenza viene aggiunta parzialmente
                    # Questo potrebbe causare problemi o mascherarli.
                    break  # Interrompe il loop interno per questa sequenza
            # Aggiunge tmp_seq a res, anche se incompleta in caso di OOV
            res.append(tmp_seq)
        return res


# Funzione originale collate_fn
def collate_fn(data, pad_token, DEVICE):  # Accetta pad_token e DEVICE
    # Funzione interna originale merge
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Usa pad_token (che è l'ID) per riempire
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        # Rimosso detach() perché non necessario se si crea nuovo tensore
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    # Sposta su DEVICE e calcola numero token
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item
