
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

# ==============================================================================
# --- LSTM Model for Joint Intent Classification and Slot Filling ---
# Supports configurable bidirectionality and dropout.
# ==============================================================================

class ModelIAS(nn.Module):
    """
    LSTM-based model for joint Intent Classification and Slot Filling.
    """
    def __init__(
        self,
        hidden_size,      # LSTM hidden state size (per direction)
        num_slot_classes,
        num_intent_classes,
        embedding_size,
        vocab_size,
        num_lstm_layers=1,
        pad_token_idx=0,
        is_bidirectional=False,
        dropout_p=0.0,
    ):
        super(ModelIAS, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_idx)

        direction_multiplier = 2 if is_bidirectional else 1
        linear_input_size = hidden_size * direction_multiplier

        self.lstm_encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=is_bidirectional,
            batch_first=True,
        )
        self.is_bidirectional_lstm = is_bidirectional

        self.slot_classifier = nn.Linear(linear_input_size, num_slot_classes)
        self.intent_classifier = nn.Linear(linear_input_size, num_intent_classes)

        self.dropout_layer = nn.Dropout(dropout_p)
        self.dropout_probability = dropout_p

    def forward(self, input_utterances, sequence_lengths):
        """
        Defines the forward pass of the model.
        Args:
            input_utterances (Tensor): Token IDs (batch_size, seq_len).
            sequence_lengths (Tensor): Actual lengths of utterances in the batch.
        Returns:
            slot_logits (Tensor): (batch_size, num_slot_classes, seq_len).
            intent_logits (Tensor): (batch_size, num_intent_classes).
        """
        embedded_utterances = self.embedding_layer(input_utterances)

        if self.dropout_probability > 0:
            embedded_utterances = self.dropout_layer(embedded_utterances)

        # Ensure lengths are positive for pack_padded_sequence
        lengths_for_packing_cpu = sequence_lengths.cpu().clone().numpy()
        if not np.all(lengths_for_packing_cpu > 0): # Defensive check for zero lengths
            lengths_for_packing_cpu = np.maximum(lengths_for_packing_cpu, 1)
        
        packed_embeddings = pack_padded_sequence(
            embedded_utterances,
            lengths_for_packing_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output_lstm, (hidden_state_lstm, _) = self.lstm_encoder(packed_embeddings)
        lstm_output_padded, _ = pad_packed_sequence(packed_output_lstm, batch_first=True)

        if self.dropout_probability > 0:
            lstm_output_padded = self.dropout_layer(lstm_output_padded)

        # Prepare LSTM output for Intent Classification
        if self.is_bidirectional_lstm:
            # Concatenate last hidden states of forward and backward LSTM
            last_hidden_for_intent = torch.cat(
                (hidden_state_lstm[-2, :, :], hidden_state_lstm[-1, :, :]), dim=1
            )
        else:
            last_hidden_for_intent = hidden_state_lstm[-1, :, :]

        if self.dropout_probability > 0:
            last_hidden_for_intent = self.dropout_layer(last_hidden_for_intent)

        # Generate Logits
        slot_logits = self.slot_classifier(lstm_output_padded)
        intent_logits = self.intent_classifier(last_hidden_for_intent)

        # Permute Slot Logits for CrossEntropyLoss: (N, C, seq_len)
        slot_logits = slot_logits.permute(0, 2, 1)

        return slot_logits, intent_logits
