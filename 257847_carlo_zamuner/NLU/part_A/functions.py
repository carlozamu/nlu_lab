# --- START OF FILE functions.py ---
# Contains model weight initialization, training loop, and evaluation loop functions.

import torch
import torch.nn as nn

# import torch.nn.functional as F # Not used in this version
import numpy as np

# from tqdm import tqdm # tqdm is typically used in the main script's epoch loop
from sklearn.metrics import classification_report

try:
    from conll import evaluate  # CoNLL evaluation script
except ImportError:
    print("CRITICAL ERROR in functions.py: 'conll.py' not found.")
    # Or re-raise to be caught by the main script if preferred
    # raise ImportError("conll.py not found, needed for evaluation.")

# ==============================================================================
# --- Weight Initialization Function ---
# ==============================================================================


def init_weights(model_module):
    """
    Initializes weights of RNN and Linear layers in the model.
    Args:
        model_module (nn.Module): The model to initialize.
    """
    for module in model_module.modules():
        if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in module.named_parameters():
                if "weight_ih" in name:  # Input-to-hidden weights
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:  # Hidden-to-hidden weights
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)  # Biases to zero
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


# ==============================================================================
# --- Training Loop (for LSTM-based Intent/Slot model) ---
# ==============================================================================


def train_loop(
    data_loader, optimizer, criterion_slots, criterion_intents, model, clip_value=5.0
):
    """
    Executes one training epoch for the LSTM-based joint model.
    Args:
        data_loader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        criterion_slots (Loss): Loss function for slots.
        criterion_intents (Loss): Loss function for intents.
        model (nn.Module): The model to train.
        clip_value (float): Gradient clipping value.
    Returns:
        list: List of batch losses for the epoch.
    """
    model.train()
    epoch_batch_losses = []
    for batch_sample in data_loader:  # Renamed 'sample' to 'batch_sample' for clarity
        optimizer.zero_grad()
        # ModelIAS expects "utterances" and "slots_len" from the collate_fn output
        slot_logits, intent_logits = model(
            batch_sample["utterances"], batch_sample["slots_len"]
        )

        loss_intent = criterion_intents(intent_logits, batch_sample["intents"])
        loss_slot = criterion_slots(
            slot_logits, batch_sample["y_slots"]
        )  # Assumes slot_logits is (N, C, L)

        current_loss = loss_intent + loss_slot
        epoch_batch_losses.append(current_loss.item())

        current_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
    return epoch_batch_losses  # Returns list of losses, main script can average it


# ==============================================================================
# --- Evaluation Loop (for LSTM-based Intent/Slot model) ---
# ==============================================================================


def eval_loop(data_loader, criterion_slots, criterion_intents, model, lang_obj):
    """
    Performs evaluation for the LSTM-based joint model.
    Args:
        data_loader (DataLoader): DataLoader for the evaluation set.
        criterion_slots (Loss): Loss function for slots.
        criterion_intents (Loss): Loss function for intents.
        model (nn.Module): The model to evaluate.
        lang_obj (Lang): The Lang object for converting IDs to labels/words.
    Returns:
        tuple: (slot_conll_results, intent_class_report, average_epoch_loss)
    """
    model.eval()
    epoch_batch_losses = []
    all_ref_intents, all_hyp_intents = [], []
    all_ref_slot_sequences, all_hyp_slot_sequences = [], []

    with torch.no_grad():
        for batch_sample in data_loader:  # Renamed 'sample' to 'batch_sample'
            slot_logits, intent_logits = model(
                batch_sample["utterances"], batch_sample["slots_len"]
            )

            loss_intent = criterion_intents(intent_logits, batch_sample["intents"])
            loss_slot = criterion_slots(slot_logits, batch_sample["y_slots"])
            current_loss = loss_intent + loss_slot
            epoch_batch_losses.append(current_loss.item())

            # Intent inference
            pred_intent_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            all_hyp_intents.extend(
                [lang_obj.id2intent.get(idx, "UNK_INTENT") for idx in pred_intent_ids]
            )
            all_ref_intents.extend(
                [
                    lang_obj.id2intent.get(idx, "UNK_INTENT")
                    for idx in batch_sample["intents"].cpu().tolist()
                ]
            )

            # Slot inference
            # slot_logits from ModelIAS is (batch, num_slot_classes, seq_len)
            predicted_slot_ids_batch = torch.argmax(
                slot_logits, dim=1
            )  # (batch, seq_len)

            for i in range(
                predicted_slot_ids_batch.size(0)
            ):  # Index for samples in the batch
                current_seq_length = batch_sample["slots_len"][i].item()

                # Get original word IDs for the current sample's utterance
                # ** THIS IS THE CORRECTED PART **
                utterance_word_ids = (
                    batch_sample["utterances"][i][:current_seq_length].cpu().tolist()
                )

                # Convert word IDs to actual words
                original_words = [
                    lang_obj.id2word.get(word_id, lang_obj.word2id.get("unk", "UNK"))
                    for word_id in utterance_word_ids
                ]

                # Ground truth slot labels
                true_slot_ids = (
                    batch_sample["y_slots"][i][:current_seq_length].cpu().tolist()
                )
                true_slot_labels = [
                    lang_obj.id2slot.get(
                        slot_id, lang_obj.slot2id.get("pad", "UNK_SLOT")
                    )
                    for slot_id in true_slot_ids
                ]

                # Predicted slot labels
                pred_slot_ids_sample_seq = (
                    predicted_slot_ids_batch[i][:current_seq_length].cpu().tolist()
                )
                pred_slot_labels = [
                    lang_obj.id2slot.get(
                        slot_id, lang_obj.slot2id.get("pad", "UNK_SLOT")
                    )
                    for slot_id in pred_slot_ids_sample_seq
                ]

                # Ensure consistent lengths for zipping, though ideally they should match
                min_len_ref = min(len(original_words), len(true_slot_labels))
                all_ref_slot_sequences.append(
                    list(
                        zip(
                            original_words[:min_len_ref], true_slot_labels[:min_len_ref]
                        )
                    )
                )

                min_len_hyp = min(len(original_words), len(pred_slot_labels))
                all_hyp_slot_sequences.append(
                    list(
                        zip(
                            original_words[:min_len_hyp], pred_slot_labels[:min_len_hyp]
                        )
                    )
                )

    slot_conll_results = {"total": {"f": 0.0}}  # Default in case of error
    if all_ref_slot_sequences and all_hyp_slot_sequences:
        try:
            slot_conll_results = evaluate(
                all_ref_slot_sequences, all_hyp_slot_sequences
            )
        except Exception as ex:
            # Use tqdm.write if tqdm is available and active in the calling scope (main.py epoch loop)
            # For a library function, print might be acceptable or re-raise the exception.
            print(f"Warning during CoNLL evaluation in eval_loop: {ex}")
            # Optional: print mismatched sets if useful for debugging
            # ref_s = set(token for seq in all_ref_slot_sequences for _, token in seq)
            # hyp_s = set(token for seq in all_hyp_slot_sequences for _, token in seq)
            # if hyp_s.difference(ref_s): print(f"CoNLL warning - Hyp slots not in Ref: {hyp_s.difference(ref_s)}")

    intent_class_report = classification_report(
        all_ref_intents, all_hyp_intents, zero_division=0, output_dict=True
    )

    average_epoch_loss = (
        np.mean(epoch_batch_losses) if epoch_batch_losses else float("inf")
    )

    return slot_conll_results, intent_class_report, average_epoch_loss


# --- END OF FILE functions.py ---
