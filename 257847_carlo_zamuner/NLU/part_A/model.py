import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
JOINT INTENT AND SLOT CLASSIFICATION MODEL
------------------------------------------
This module implements a bidirectional LSTM-based model for joint Intent and Slot Classification.

Architecture:
1. Word Embedding Layer
2. Bidirectional LSTM Encoder
3. Two output heads:
   - Slot Classification: sequence labeling (one label per word)
   - Intent Classification: sentence-level classification

The model shares the LSTM encoder between both tasks, enabling joint learning.
"""

class ModelIAS(nn.Module):
    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=1,
        pad_index=0,
        bidirectional=True,
        dropout=0.1,
    ):
        super(ModelIAS, self).__init__()

        # Word embedding layer - converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Bidirectional LSTM encoder - processes the sequence in both directions
        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True
        )

        # Calculate LSTM output size (double if bidirectional)
        lstm_output_size = hid_size * 2 if bidirectional else hid_size

        # Output layers for the two tasks
        self.slot_out = nn.Linear(lstm_output_size, out_slot)    # Slot classification head
        self.intent_out = nn.Linear(lstm_output_size, out_int)   # Intent classification head
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, utterance, seq_lengths):
        """
        Forward pass for joint Intent and Slot Classification
        
        Args:
            utterance: Batch of tokenized utterances [batch_size, max_len]
            seq_lengths: Actual lengths of each utterance [batch_size]
            
        Returns:
            slots: Slot predictions [batch_size, num_slots, max_len] 
            intent: Intent predictions [batch_size, num_intents]
        """
        
        # 1. Convert word indices to embeddings
        utt_emb = self.embedding(utterance)
        utt_emb = self.dropout(utt_emb)  # Apply dropout after embedding

        # 2. Pack sequences for efficient LSTM processing (handles variable lengths)
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        
        # 3. Process through bidirectional LSTM
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # 4. Unpack sequences back to padded format
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)

        # 5. Prepare intent representation from final hidden states
        if self.utt_encoder.bidirectional:
            # Concatenate forward and backward final hidden states
            last_hidden = torch.cat(
                (last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1
            )
        else:
            # Use only the final hidden state
            last_hidden = last_hidden[-1, :, :]

        last_hidden = self.dropout(last_hidden)

        # 6. Generate predictions for both tasks
        slots = self.slot_out(utt_encoded)    # Sequence labeling: one prediction per timestep
        intent = self.intent_out(last_hidden)  # Sentence classification: one prediction per utterance

        # 7. Permute slots for CrossEntropyLoss format [batch_size, num_classes, sequence_length]
        slots = slots.permute(0, 2, 1)
        
        return slots, intent