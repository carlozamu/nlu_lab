import torch
import torch.nn as nn


class LM_LSTM_Base(nn.Module):
    """
    Basic LSTM Language Model without regularization techniques.
    Used for Part 1A - Step 1: Replace RNN with LSTM
    """

    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1,
    ):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output


class LM_LSTM_With_Dropout(nn.Module):
    """
    LSTM Language Model with dropout regularization.
    Used for Part 1A - Step 2: Add two dropout layers
    Used for Part 1A - Step 3: Add two dropout layers + replace SGD with AdamW

    Architecture: Embedding -> Dropout -> LSTM -> Dropout -> Linear -> Output
    """

    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: float = 0.1,
        fc_dropout: float = 0.1,
        n_layers: int = 1,
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        # Word embeddings with dropout
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)

        # LSTM processing
        lstm_out, _ = self.lstm(emb_dropped)

        # Output projection with dropout
        out_dropped = self.dropout_fc(lstm_out)
        output_logits = self.output(out_dropped)

        # Permute for CrossEntropyLoss: (batch, vocab, seq_len)
        output = output_logits.permute(0, 2, 1)
        return output
