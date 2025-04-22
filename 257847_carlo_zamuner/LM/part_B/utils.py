import torch
import torch.utils.data as data


# Legge un file di corpus, aggiunge <eos> a ogni riga.
def read_file(path, eos_token="<eos>"):
    """Legge righe da un file, aggiunge eos_token e ritorna lista di stringhe."""
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


# Crea un vocabolario (word -> id) da un corpus.
def get_vocab(corpus, special_tokens=[]):
    """Crea un dizionario word-to-id dal corpus e token speciali."""
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


# Classe per gestire vocabolario (word <-> id).
class Lang:
    """Gestisce il mapping tra parole e indici."""

    def __init__(self, corpus, special_tokens=[]):
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}


# Dataset per Penn Treebank (PTB).
class PennTreeBank(data.Dataset):
    """Dataset PyTorch per PTB, crea coppie source/target."""

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

    # Metodo privato per mappare sequenze a ID
    def _map_sequences(self, sequences_of_tokens, word2id_dict):
        """Mappa liste di sequenze di token ai loro ID."""
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
                    f"Attenzione: OOV trovato ma <unk> non nel vocabolario! Sequenza saltata: {' '.join(seq)}"
                )
                continue
            mapped_sequences.append(ids)
        return [seq for seq in mapped_sequences if seq]  # Rimuove eventuali seq vuote


# Funzione Collate per DataLoader - Padda le sequenze nel batch.
def collate_fn(batch_data, pad_token_id, device):
    """Prepara batch: ordina, padda, crea tensori su device, calcola n_tokens."""
    batch_data.sort(key=lambda x: len(x["source"]), reverse=True)
    sources = [item["source"] for item in batch_data]
    targets = [item["target"] for item in batch_data]

    def _pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        padded = torch.LongTensor(len(sequences), max_len).fill_(pad_token_id)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, lengths

    padded_sources, source_lengths = _pad_sequences(sources)
    padded_targets, target_lengths = _pad_sequences(targets)

    collated_batch = {
        "source": padded_sources.to(device),
        "target": padded_targets.to(device),
        "number_tokens": sum(target_lengths),
    }
    return collated_batch
