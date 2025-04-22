# --- START OF FILE model.py ---

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np  # Importato per gestire le lunghezze numpy

# Qui è dove definisci l'architettura del tuo modello usando PyTorch
# QUESTA È LA VERSIONE BASELINE (LSTM Unidirezionale, No Dropout)


class ModelIAS(nn.Module):
    """Modello BASELINE per Intent And Slot filling basato su LSTM UNIDIREZIONALE."""

    # Rimossi bidirectional e dropout_rate dagli argomenti __init__
    def __init__(
        self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0
    ):
        """
        Inizializza i layer del modello BASELINE.
        Args:
            hid_size (int): Dimensione dello stato nascosto dell'LSTM.
            out_slot (int): Numero di classi di slot.
            out_int (int): Numero di classi di intent.
            emb_size (int): Dimensione degli embedding delle parole.
            vocab_len (int): Dimensione del vocabolario.
            n_layer (int): Numero di layer LSTM (default: 1).
            pad_index (int): Indice usato per il padding (default: 0).
        """
        super(ModelIAS, self).__init__()
        # Layer di Embedding
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Layer LSTM - Forzato a essere UNIDIREZIONALE
        # bidirectional=False è il default, ma lo esplicitiamo per chiarezza
        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=False, batch_first=True
        )

        # Layer Lineari di Output - L'input è hid_size (non hid_size*2)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

        # NESSUN layer Dropout definito qui per la baseline

    def forward(self, utterance, seq_lengths):
        """
        Passaggio forward del modello BASELINE.
        Args:
            utterance (Tensor): Batch di sequenze di indici (batch_size, seq_len).
            seq_lengths (Tensor): Lunghezze originali delle sequenze.
        Returns:
            tuple: (slots, intent)
                   - slots (Tensor): Logits slot (batch_size, num_slot_classes, seq_len).
                   - intent (Tensor): Logits intent (batch_size, num_intent_classes).
        """
        # 1. Embedding
        utt_emb = self.embedding(utterance)
        # NESSUN dropout applicato qui

        # 2. Packing
        lengths_cpu = seq_lengths.cpu().clone().numpy()
        if not np.all(lengths_cpu > 0):
            # print("Attenzione: rilevate lunghezze di sequenza pari a zero nel batch.") # Commentato per meno output
            lengths_cpu = np.maximum(lengths_cpu, 1)
        packed_input = pack_padded_sequence(
            utt_emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # 3. Processamento RNN (LSTM Unidirezionale)
        packed_output, (hidden, cell) = self.utt_encoder(packed_input)

        # 4. Unpacking
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        # NESSUN dropout applicato qui

        # 5. Estrazione ultimo stato nascosto per intento (solo ultima direzione)
        # hidden è (n_layer, batch, hid_size) perché unidirezionale
        last_hidden = hidden[-1, :, :]
        # NESSUN dropout applicato qui

        # 6. Calcolo Logits
        slots = self.slot_out(utt_encoded)  # Input è (batch, seq_len, hid_size)
        intent = self.intent_out(last_hidden)  # Input è (batch, hid_size)

        # 7. Permutazione Dimensioni Slot per Loss
        slots = slots.permute(0, 2, 1)

        return slots, intent


# --- END OF FILE model.py ---
