import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from conll import evaluate

"""
TRAINING AND EVALUATION FUNCTIONS
---------------------------------
This module contains functions for:
1. Model weight initialization
2. Training loop with joint loss
3. Evaluation loop with slot F1 and intent accuracy metrics
"""

def init_weights(mat):
    """
    Custom weight initialization for LSTM and Linear layers
    Uses Xavier uniform for input-hidden weights and orthogonal for hidden-hidden weights
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            # Special initialization for RNN layers
            for name, param in m.named_parameters():
                if "weight_ih" in name:  # Input-to-hidden weights
                    for idx in range(4):  # 4 gates in LSTM
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:  # Hidden-to-hidden weights
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:  # Bias terms
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                # Linear layer initialization
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Single training epoch for joint Intent and Slot Classification
    
    Args:
        data: Training data loader
        optimizer: PyTorch optimizer
        criterion_slots: Loss function for slot classification
        criterion_intents: Loss function for intent classification
        model: The neural network model
        clip: Gradient clipping threshold
        
    Returns:
        loss_array: List of batch losses for this epoch
    """
    model.train()  # Set model to training mode
    loss_array = []
    
    for sample in data:
        optimizer.zero_grad()  # Clear gradients
        
        # Forward pass
        slots, intent = model(sample["utterances"], sample["slots_len"])
        
        # Calculate losses for both tasks
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        
        # Joint loss: simple sum of both losses
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluation loop for both validation and test sets
    
    Args:
        data: Evaluation data loader
        criterion_slots: Loss function for slot classification
        criterion_intents: Loss function for intent classification  
        model: The neural network model
        lang: Language object for converting indices back to labels
        
    Returns:
        results: Slot classification results (including F1 score)
        report_intent: Intent classification results (including accuracy)
        loss_array: List of batch losses
    """
    model.eval()  # Set model to evaluation mode
    loss_array = []

    # Store predictions and ground truth for both tasks
    ref_intents = []  # True intent labels
    hyp_intents = []  # Predicted intent labels

    ref_slots = []    # True slot sequences  
    hyp_slots = []    # Predicted slot sequences
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for sample in data:
            # Forward pass
            slots, intents = model(sample["utterances"], sample["slots_len"])
            
            # Calculate losses
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            # Process intent predictions
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Process slot predictions (more complex due to variable sequence lengths)
            output_slots = torch.argmax(slots, dim=1)
            
            for id_seq, seq in enumerate(output_slots):
                # Get actual sequence length (without padding)
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterances"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                
                # Convert indices back to labels
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                # Create (word, slot_label) pairs for CoNLL evaluation format
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
    # Calculate slot classification metrics using CoNLL evaluation
    try:
        results = evaluate(ref_slots, hyp_slots)  # Returns F1, precision, recall
    except Exception:
        results = {"total": {"f": 0}}  # Fallback if evaluation fails

    # Calculate intent classification metrics using sklearn
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    
    return results, report_intent, loss_array