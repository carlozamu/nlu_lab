import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    """
    Variational Dropout (also called Locked Dropout).
    Unlike standard dropout, this applies the same dropout mask across all time steps,
    which is more effective for RNNs and prevents overfitting better.
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip dropout during evaluation or if rate is 0
        if not self.training or self.dropout_rate == 0.0:
            return x

        # Create mask shape: (batch_size, 1, feature_size)
        # The '1' dimension ensures same mask is applied across all time steps
        mask_shape = (x.size(0), 1, x.size(2))
        keep_probability = 1.0 - self.dropout_rate

        # Generate binary mask using Bernoulli distribution
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_probability, device=x.device, dtype=x.dtype)
        )

        # Scale mask to maintain expected value during training
        mask = mask / keep_probability
        return x * mask


class LM_LSTM_Base(nn.Module):
    """
    Basic LSTM Language Model without any regularization techniques.
    This serves as a baseline implementation.
    """
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, n_layers: int = 1
    ):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        # Permute to (batch_size, vocab_size, sequence_length) for CrossEntropyLoss
        output = output_logits.permute(0, 2, 1)
        return output

class LM_LSTM_With_Dropout(nn.Module):
    """
    LSTM Language Model with standard dropout regularization.
    Applies dropout after embedding and before the final linear layer.
    """
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, emb_dropout: float = 0.1, fc_dropout: float = 0.1,
        n_layers: int = 1
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout) # Dropout after embedding
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)   # Dropout before final linear layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb_dropped)
        out_dropped = self.dropout_fc(lstm_out)
        output_logits = self.output(out_dropped)
        output = output_logits.permute(0, 2, 1)
        return output


class LM_LSTM_WeightTying(nn.Module):
    """
    LSTM Language Model with Weight Tying technique.
    Weight tying shares parameters between input embedding and output projection layers,
    reducing model parameters and often improving performance.
    """
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, n_layers: int = 1
    ):
        super(LM_LSTM_WeightTying, self).__init__()
        # Weight tying requires embedding and hidden dimensions to be equal
        assert emb_size == hidden_size
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Key technique: share weights between embedding and output layers
        self.output.weight = self.embedding.weight 

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output


class LM_LSTM_VarDropWT(nn.Module):
    """
    Advanced LSTM Language Model combining multiple state-of-the-art techniques:
    1. Variational Dropout (LockedDropout) - more effective regularization for RNNs
    2. Weight Tying - parameter sharing between embedding and output layers
    
    This model represents the final, most sophisticated version used in the experiments.
    """

    def __init__(
        self,
        emb_size: int,      # Embedding dimension
        hidden_size: int,   # LSTM hidden dimension  
        output_size: int,   # Vocabulary size
        pad_index: int = 0, # Padding token index
        n_layers: int = 1,  # Number of LSTM layers
        dropoute: float = 0.1, # Variational dropout rate for embeddings
        dropouto: float = 0.4  # Variational dropout rate for output
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        
        # Weight tying constraint: embedding and hidden sizes must match
        assert emb_size == hidden_size, \
            "For this Weight Tying setup, embedding_size must equal hidden_size."

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # LSTM with no built-in dropout (we use custom variational dropout)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True, dropout=0
        )
        self.output = nn.Linear(hidden_size, output_size)

        # Variational dropout layers - these maintain consistent masks across time steps
        self.lockdrop_e = LockedDropout(dropoute)  # For embedding layer
        self.lockdrop_o = LockedDropout(dropouto)  # For output layer

        # Weight tying: share parameters between embedding input and output projection
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Standard embedding lookup
        emb = self.embedding(input_sequence)            
        # Apply variational dropout to embeddings
        emb_dropped = self.lockdrop_e(emb)              

        # LSTM forward pass
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden)

        # Apply variational dropout to LSTM outputs
        lstm_out_dropped = self.lockdrop_o(lstm_out)

        # Final linear transformation (uses tied weights from embedding)
        output_logits = self.output(lstm_out_dropped)

        # Reshape for loss computation
        output = output_logits.permute(0, 2, 1)
        return output, hidden_out