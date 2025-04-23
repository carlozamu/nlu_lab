# --- START OF FILE model.py ---

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

# Qui è dove definisci l'architettura del tuo modello usando PyTorch
# MODELLO CON BIDIREZIONALITÀ E DROPOUT CONFIGURABILI


class ModelIAS(nn.Module):
    """Modello per Intent And Slot filling basato su LSTM con bidirezionalità e dropout configurabili."""

    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        bidirectional=False,
        dropout_rate=0.0,
    ):
        """
        Inizializza i layer del modello.
        Args:
            hid_size (int): Dimensione dello stato nascosto dell'LSTM (per *ogni* direzione).
            out_slot (int): Numero di classi di slot.
            out_int (int): Numero di classi di intent.
            emb_size (int): Dimensione degli embedding.
            vocab_len (int): Dimensione vocabolario.
            n_layer (int): Numero di layer LSTM (default: 1).
            pad_index (int): Indice padding (default: 0).
            bidirectional (bool): Se True, usa LSTM bidirezionale (default: False).
            dropout_rate (float): Tasso di dropout (default: 0.0).
        """
        super(ModelIAS, self).__init__()
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # --- Configurazione LSTM ---
        lstm_hid_size = hid_size
        direction_multiplier = 2 if bidirectional else 1
        linear_in_size = lstm_hid_size * direction_multiplier

        # Layer LSTM con bidirezionalità configurabile
        self.utt_encoder = nn.LSTM(
            emb_size,
            lstm_hid_size,
            n_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.is_bidirectional = bidirectional

        # Layer Lineari di Output - Adattano la dimensione di input
        self.slot_out = nn.Linear(linear_in_size, out_slot)
        self.intent_out = nn.Linear(linear_in_size, out_int)

        # Layer Dropout (presente ma applicato solo se dropout_rate > 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, utterance, seq_lengths):
        """
        Passaggio forward del modello.
        """
        utt_emb = self.embedding(utterance)

        # Applica dropout agli embedding (solo se dropout_rate > 0)
        if self.dropout_rate > 0:
            utt_emb = self.dropout(utt_emb)

        # Packing
        lengths_cpu = seq_lengths.cpu().clone().numpy()
        if not np.all(lengths_cpu > 0):
            lengths_cpu = np.maximum(lengths_cpu, 1)
        packed_input = pack_padded_sequence(
            utt_emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        # Processamento RNN (LSTM)
        packed_output, (hidden, cell) = self.utt_encoder(packed_input)

        # Unpacking
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Applica dropout all'output dell'LSTM (solo se dropout_rate > 0)
        if self.dropout_rate > 0:
            utt_encoded = self.dropout(utt_encoded)

        # Estrazione ultimo stato nascosto per intento
        # hidden è (n_layer * num_directions, batch, hid_size)
        if self.is_bidirectional:
            # Concatena ultimo stato forward e backward dell'ultimo layer
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Prendi ultimo stato dell'ultimo layer
            last_hidden = hidden[-1, :, :]

        # Applica dropout all'ultimo stato nascosto (solo se dropout_rate > 0)
        if self.dropout_rate > 0:
            last_hidden = self.dropout(last_hidden)

        # Calcolo Logits
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)

        # Permutazione Dimensioni Slot per Loss
        slots = slots.permute(0, 2, 1)

        return slots, intent


# --- END OF FILE model.py ---
