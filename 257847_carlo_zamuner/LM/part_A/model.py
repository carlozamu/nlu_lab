import torch
import torch.nn as nn


class LM_LSTM_With_Dropout(nn.Module):
    """
    LSTM Language Model with dropout regularization.
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
        n_layers: int = 1
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

class LM_LSTM_Base(nn.Module):
    """
    Basic LSTM Language Model without regularization techniques.
    Simple baseline architecture.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1
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

class LockedDropout(nn.Module):
    """
    Variational dropout: applies the same dropout mask across all time steps.
    This maintains consistency in dropped features through the sequence.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0:
            return x

        # Create mask for entire sequence (same mask for all time steps)
        mask_shape = (x.size(0), 1, x.size(2))
        keep_prob = 1.0 - self.dropout
        mask = torch.bernoulli(torch.full(mask_shape, keep_prob, device=x.device, dtype=x.dtype))
        mask = mask / keep_prob
        return x * mask

class LM_LSTM_WeightTying(nn.Module):
    """
    LSTM with Weight Tying: embedding and output weights are shared.
    This reduces parameters and can improve performance when emb_size == hidden_size.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1
    ):
        super(LM_LSTM_WeightTying, self).__init__()
        assert emb_size == hidden_size, "Weight Tying in this setup requires embedding_size == hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Share weights between embedding and output layer
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)   
        lstm_out, _ = self.lstm(emb)           
        output_logits = self.output(lstm_out)  
        output = output_logits.permute(0, 2, 1) 
        return output

class LM_LSTM_VarDropWT(nn.Module):
    """
    Advanced LSTM with Variational Dropout + Weight Tying.
    Combines locked dropout and weight sharing for better regularization.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1,
        dropoute: float = 0.1,  # Embedding dropout
        dropouto: float = 0.4   # Output dropout
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        assert emb_size == hidden_size, "Weight Tying requires embedding_size == hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=0)
        self.output = nn.Linear(hidden_size, output_size)

        # Variational dropout layers
        self.lockdrop_e = LockedDropout(dropoute)
        self.lockdrop_o = LockedDropout(dropouto)

        # Weight tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor, hidden: tuple = None) -> tuple[torch.Tensor, tuple]:
        emb = self.embedding(input_sequence)          
        emb_dropped = self.lockdrop_e(emb)             # Variational dropout on embeddings
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden) 
        lstm_out_dropped = self.lockdrop_o(lstm_out)    # Variational dropout on LSTM output
        output_logits = self.output(lstm_out_dropped)   
        output = output_logits.permute(0, 2, 1)         
        return output, hidden_out