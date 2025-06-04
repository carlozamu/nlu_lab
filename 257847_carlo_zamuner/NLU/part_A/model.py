import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True
        )

        lstm_output_size = hid_size * 2 if bidirectional else hid_size

        self.slot_out = nn.Linear(lstm_output_size, out_slot)
        self.intent_out = nn.Linear(lstm_output_size, out_int)
        self.dropout = nn.Dropout(dropout)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        utt_emb = self.dropout(utt_emb)  # Apply dropout after embedding

        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True
        )
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)

        if self.utt_encoder.bidirectional:
            last_hidden = torch.cat(
                (last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1
            )
        else:
            last_hidden = last_hidden[-1, :, :]

        last_hidden = self.dropout(last_hidden)

        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0, 2, 1)
        return slots, intent
