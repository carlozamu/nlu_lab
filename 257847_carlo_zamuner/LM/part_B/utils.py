import torch
import torch.utils.data as data


# Legge corpus
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


# Crea vocabolario
def get_vocab(corpus, special_tokens=[]):
    output = {}
    idx = 0
    for token in special_tokens:
        if token not in output:
            output[token] = idx
            idx += 1
    for sentence in corpus:
        for word in sentence.split():
            if word not in output:
                output[word] = idx
                idx += 1
    return output


# Gestore vocabolario
class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}


# Dataset PTB
class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        sources_tokens = []
        targets_tokens = []
        for sentence in corpus:
            tokens = sentence.split()
            if len(tokens) > 1:
                sources_tokens.append(tokens[:-1])
                targets_tokens.append(tokens[1:])
        self.source_ids = self._map_sequences(sources_tokens, lang.word2id)
        self.target_ids = self._map_sequences(targets_tokens, lang.word2id)

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {"source": src, "target": trg}

    def _map_sequences(self, sequences_of_tokens, word2id_dict):
        mapped_sequences = []
        unk_token_id = word2id_dict.get("<unk>")
        for seq in sequences_of_tokens:
            ids = [
                word2id_dict.get(
                    token, unk_token_id if unk_token_id is not None else -1
                )
                for token in seq
            ]
            if unk_token_id is None and any(idx == -1 for idx in ids):
                print(
                    f"Attenzione: OOV trovato e <unk> non nel vocabolario! Sequenza saltata."
                )
                continue
            mapped_sequences.append(ids)
        return [seq for seq in mapped_sequences if seq]


# Funzione Collate
def collate_fn(batch_data, pad_token_id, device):
    batch_data.sort(key=lambda x: len(x["source"]), reverse=True)
    sources = [item["source"] for item in batch_data]
    targets = [item["target"] for item in batch_data]

    def _pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        padded = torch.LongTensor(len(sequences), max_len).fill_(pad_token_id)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq[: len(seq)]  # Usa len(seq)
        return padded, lengths

    padded_sources, _ = _pad_sequences(sources)  # Non servono source_lengths
    padded_targets, target_lengths = _pad_sequences(targets)

    return {
        "source": padded_sources.to(device),
        "target": padded_targets.to(device),
        "number_tokens": sum(target_lengths),
    }
