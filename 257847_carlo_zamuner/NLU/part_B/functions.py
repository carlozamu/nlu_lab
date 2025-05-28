import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from conll import evaluate


def init_weights_rnn(model_module):
    """Initializes weights of RNN and Linear layers."""
    for module in model_module.modules():
        if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in module.named_parameters():  # type: ignore
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


def train_loop_bert(
    data_loader, optimizer, criterion_slots, criterion_intents, model, grad_clip=1.0
):
    """Executes one training epoch for the BERT model."""
    model.train()
    total_epoch_loss = 0.0
    for batch in tqdm(data_loader, desc="Training BERT", leave=False, unit="batch"):
        optimizer.zero_grad()
        intent_logits, slot_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
        loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch["slot_labels"])
        current_loss = loss_intent + loss_slot
        total_epoch_loss += current_loss.item()
        current_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return total_epoch_loss / len(data_loader) if data_loader else float("inf")


def eval_loop_bert(
    data_loader, criterion_slots, criterion_intents, model, lang_obj, tokenizer_obj=None
):  # tokenizer_obj can be passed if needed by eval logic
    """Performs evaluation for the BERT model, handling slot alignment."""
    model.eval()
    total_epoch_loss = 0.0
    all_ref_intents, all_hyp_intents = [], []
    all_ref_slot_sequences, all_hyp_slot_sequences = [], []

    with torch.no_grad():
        for batch in tqdm(
            data_loader, desc="Evaluating BERT", leave=False, unit="batch"
        ):
            intent_logits, slot_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
            loss_slot = criterion_slots(
                slot_logits.permute(0, 2, 1), batch["slot_labels"]
            )
            total_epoch_loss += (loss_intent + loss_slot).item()

            pred_intent_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            all_hyp_intents.extend(
                [lang_obj.id2intent.get(idx, "UNK_INTENT") for idx in pred_intent_ids]
            )
            all_ref_intents.extend(
                [
                    lang_obj.id2intent.get(idx, "UNK_INTENT")
                    for idx in batch["intent_labels"].cpu().tolist()
                ]
            )

            pred_slot_ids_batch = torch.argmax(slot_logits, dim=-1).cpu().tolist()
            for i in range(batch["input_ids"].size(0)):
                word_ids_sample = batch["word_ids"][i]
                attn_mask_sample = batch["attention_mask"][i].cpu().tolist()
                original_slots_sample = batch["original_slot_labels"][i]
                original_words_sample = batch["original_words"][i]
                pred_slot_ids_sample = pred_slot_ids_batch[i]
                aligned_pred_slots = []
                last_word_idx = -1
                for token_idx, word_idx in enumerate(word_ids_sample):
                    if word_idx is not None and attn_mask_sample[token_idx] == 1:
                        if word_idx != last_word_idx:
                            if word_idx < len(original_words_sample):
                                aligned_pred_slots.append(
                                    lang_obj.id2slot.get(
                                        pred_slot_ids_sample[token_idx], "O"
                                    )
                                )
                        last_word_idx = word_idx

                min_len = min(
                    len(original_words_sample),
                    len(original_slots_sample),
                    len(aligned_pred_slots),
                )
                if min_len > 0:
                    all_ref_slot_sequences.append(
                        list(
                            zip(
                                original_words_sample[:min_len],
                                original_slots_sample[:min_len],
                            )
                        )
                    )
                    all_hyp_slot_sequences.append(
                        list(
                            zip(
                                original_words_sample[:min_len],
                                aligned_pred_slots[:min_len],
                            )
                        )
                    )

    slot_f1 = 0.0
    if all_ref_slot_sequences and all_hyp_slot_sequences:
        try:
            slot_f1 = (
                evaluate(all_ref_slot_sequences, all_hyp_slot_sequences)
                .get("total", {})
                .get("f", 0.0)
            )
        except Exception:
            pass

    intent_acc = classification_report(
        all_ref_intents, all_hyp_intents, zero_division=0, output_dict=True
    ).get("accuracy", 0.0)
    mean_epoch_loss = (
        total_epoch_loss / len(data_loader) if data_loader else float("inf")
    )
    return slot_f1, intent_acc, mean_epoch_loss
