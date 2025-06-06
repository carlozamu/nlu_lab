import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import copy
from functions import (
    device,
    IGNORE_INDEX,
    Lang,
    IntentsAndSlotsBERT,
    collate_fn_bert,
    load_data,
)
from utils import train_loop_bert, eval_loop_bert
from model import JointBERTModel
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

# ========== MODE CONFIGURATION ==========
EVAL_ONLY = True

if __name__ == "__main__":
    # ========== HYPERPARAMETERS ==========
    BERT_MODELS = ["bert-large-uncased"]
    LEARNING_RATE = 5e-5
    DROPOUT_RATE = 0.3
    BATCH_SIZE_TRAIN = 128
    BATCH_SIZE_EVAL = 128
    MAX_EPOCHS = 50
    EARLY_STOP_PATIENCE = 10
    CLIP = 1.0
    MAX_SEQ_LENGTH = 50

    # ========== EVALUATION MODE ==========
    if EVAL_ONLY:
        print("\n--- EVALUATION MODE ---")

        # Check for saved models
        model_dir = "bin"
        if not os.path.exists(model_dir):
            print(f"ERROR: Model directory '{model_dir}' not found.")
            exit()

        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        if not model_files:
            print(f"ERROR: No .pt model files found in '{model_dir}'.")
            exit()

        print(f"Found {len(model_files)} model(s) to evaluate:")
        evaluation_results = []

        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            model_name = model_file
            print(f"Evaluating: {model_name}")

            try:
                # Load model checkpoint
                checkpoint = torch.load(model_path, map_location=device)

                # Data loading (needed for test set)
                data_dir = os.environ.get(
                    "ATIS_DATA_DIR", os.path.join("dataset", "ATIS")
                )
                test_raw = load_data(os.path.join(data_dir, "test.json"))

                # Tokenizer initialization
                tokenizer = BertTokenizerFast.from_pretrained(
                    checkpoint["bert_model_name"]
                )

                # Recreate language object from saved mappings
                if "intents_vocab" in checkpoint and "slots_vocab" in checkpoint:
                    lang_eval = Lang(
                        checkpoint["intents_vocab"], checkpoint["slots_vocab"]
                    )
                else:
                    # Reconstruct from saved ID mappings
                    intents_vocab_saved = set(checkpoint["intent2id"].keys())
                    slots_vocab_saved = set(checkpoint["slot2id"].keys())
                    lang_eval = Lang(intents_vocab_saved, slots_vocab_saved)

                # Create model
                if (
                    "num_intent_labels" in checkpoint
                    and "num_slot_labels" in checkpoint
                ):
                    model_eval = JointBERTModel(
                        num_intent_labels=checkpoint["num_intent_labels"],
                        num_slot_labels=checkpoint["num_slot_labels"],
                        bert_model_name=checkpoint["bert_model_name"],
                        dropout_rate=checkpoint["dropout_rate"],
                    ).to(device)
                else:
                    model_eval = JointBERTModel(
                        num_intent_labels=len(lang_eval.intent2id),
                        num_slot_labels=len(lang_eval.slot2id),
                        bert_model_name=checkpoint["bert_model_name"],
                        dropout_rate=checkpoint["dropout_rate"],
                    ).to(device)

                # Load model weights
                if "model_state_dict" in checkpoint:
                    model_eval.load_state_dict(checkpoint["model_state_dict"])
                elif "model" in checkpoint:
                    model_eval.load_state_dict(checkpoint["model"])
                else:
                    raise ValueError("No model weights found in checkpoint")

                model_eval.eval()

                # Create test dataset with saved configuration
                max_seq_len = checkpoint.get("max_seq_length", MAX_SEQ_LENGTH)
                test_dataset_eval = IntentsAndSlotsBERT(
                    test_raw, lang_eval, tokenizer, max_len=max_seq_len
                )
                test_loader_eval = DataLoader(
                    test_dataset_eval,
                    batch_size=BATCH_SIZE_EVAL,
                    collate_fn=collate_fn_bert,
                )

                # Create loss functions
                criterion_intents = nn.CrossEntropyLoss()
                criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

                # Evaluate
                slot_f1, intent_acc, test_loss = eval_loop_bert(
                    test_loader_eval,
                    criterion_slots,
                    criterion_intents,
                    model_eval,
                    lang_eval,
                    tokenizer,
                )

                evaluation_results.append(
                    {
                        "model_name": model_name,
                        "slot_f1": f"{slot_f1:.4f}",
                        "intent_accuracy": f"{intent_acc:.4f}",
                        "test_loss": f"{test_loss:.4f}",
                    }
                )
                print(f"  ✓ Success: F1={slot_f1:.4f}, Acc={intent_acc:.4f}")

            except Exception as e:
                import traceback

                print(f"  ERROR: {str(e)}")
                traceback.print_exc()
                evaluation_results.append(
                    {
                        "model_name": model_name,
                        "slot_f1": "ERROR",
                        "intent_accuracy": "ERROR",
                        "test_loss": "ERROR",
                    }
                )
                print("  ✗ Failed to evaluate")

        exit()

    # ========== TRAINING MODE ==========
    print(f"\nTraining BERT-large :")

    # ========== DATA LOADING ==========
    data_dir = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print(f"ERROR: Dataset not found at {data_dir}")
        exit()

    tmp_train_raw = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    # Create train/dev split (stratified)
    dev_portion = 0.10
    intents_full = [x["intent"] for x in tmp_train_raw]
    intent_counts = Counter(intents_full)

    strat_inputs, strat_labels, single_instance = [], [], []
    for i, intent_val in enumerate(intents_full):
        if intent_counts[intent_val] > 1:
            strat_inputs.append(tmp_train_raw[i])
            strat_labels.append(intent_val)
        else:
            single_instance.append(tmp_train_raw[i])

    if len(strat_inputs) > 1 and len(set(strat_labels)) > 1:
        train_s, dev_s, _, _ = train_test_split(
            strat_inputs,
            strat_labels,
            test_size=dev_portion,
            random_state=42,
            shuffle=True,
            stratify=strat_labels,
        )
        train_raw = train_s + single_instance
        dev_raw = dev_s
    else:
        num_dev_fb = int(len(tmp_train_raw) * dev_portion)
        if num_dev_fb > 0 and len(tmp_train_raw) - num_dev_fb > 0:
            train_raw, dev_raw = train_test_split(
                tmp_train_raw, test_size=dev_portion, random_state=42, shuffle=True
            )
        else:
            train_raw = tmp_train_raw
            dev_raw = []

    # Build vocabularies
    corpus_labels = train_raw + (dev_raw if dev_raw else []) + test_raw
    slots_vocab = set(sum([line["slots"].split() for line in corpus_labels], []))
    intents_vocab = set([line["intent"] for line in corpus_labels])

    # Create language object
    lang = Lang(intents_vocab, slots_vocab)

    # Create loss functions
    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ========== TRAIN EACH MODEL ==========
    all_results = []

    for BERT_MODEL_NAME in BERT_MODELS:
        print(f"\nTraining: {BERT_MODEL_NAME}")

        # Clear GPU cache before each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure cleanup is complete
            # Print GPU memory status for T4
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(
                f"  GPU Memory: {gpu_allocated:.1f}GB allocated, {gpu_cached:.1f}GB cached of {gpu_memory:.1f}GB total"
            )

        # Adjust batch size for BERT-large on T4 GPU (16GB VRAM)
        current_batch_size = BATCH_SIZE_TRAIN
        if "large" in BERT_MODEL_NAME.lower():
            current_batch_size = 64  # Optimized for T4 16GB
            print(f"  Using T4-optimized batch size: {current_batch_size}")

        config_name = f"BERT_Best_{BERT_MODEL_NAME.split('/')[-1]}_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{current_batch_size}"

        # Tokenizer initialization for current model
        tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

        # Create datasets
        train_dataset = IntentsAndSlotsBERT(
            train_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH
        )
        dev_dataset = (
            IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH)
            if dev_raw
            else None
        )
        test_dataset = IntentsAndSlotsBERT(
            test_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH
        )

        # Create data loaders with adjusted batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            collate_fn=collate_fn_bert,
            shuffle=True,
        )
        dev_loader = (
            DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn_bert)
            if dev_dataset
            else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, collate_fn=collate_fn_bert
        )

        try:
            # Initialize model and optimizer
            model = JointBERTModel(
                num_intent_labels=len(lang.intent2id),
                num_slot_labels=len(lang.slot2id),
                bert_model_name=BERT_MODEL_NAME,
                dropout_rate=DROPOUT_RATE,
            ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            # ========== TRAINING LOOP ==========
            best_f1_dev = 0.0
            patience_counter = EARLY_STOP_PATIENCE
            best_model_state = copy.deepcopy(model.state_dict())

            if not dev_loader:
                tqdm.write(
                    f"Fine-tuning for {MAX_EPOCHS} epochs (no dev set for early stopping)..."
                )
                for epoch in tqdm(
                    range(1, MAX_EPOCHS + 1),
                    desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]}",
                    unit="epoch",
                ):
                    train_loop_bert(
                        train_loader,
                        optimizer,
                        criterion_slots,
                        criterion_intents,
                        model,
                        CLIP,
                    )
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epoch_pbar = tqdm(
                    range(1, MAX_EPOCHS + 1),
                    desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]}",
                    unit="epoch",
                )

                for epoch_num in epoch_pbar:
                    # Training
                    train_loss = train_loop_bert(
                        train_loader,
                        optimizer,
                        criterion_slots,
                        criterion_intents,
                        model,
                        CLIP,
                    )

                    # Evaluation
                    dev_f1, dev_acc, dev_loss = eval_loop_bert(
                        dev_loader,
                        criterion_slots,
                        criterion_intents,
                        model,
                        lang,
                        tokenizer,
                    )

                    epoch_pbar.set_postfix(
                        {
                            "TrainLoss": f"{train_loss:.4f}",
                            "DevLoss": f"{dev_loss:.4f}",
                            "DevF1": f"{dev_f1:.4f}",
                            "DevAcc": f"{dev_acc:.4f}",
                        }
                    )

                    # Early stopping
                    if dev_f1 > best_f1_dev:
                        best_f1_dev = dev_f1
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = EARLY_STOP_PATIENCE
                    else:
                        patience_counter -= 1

                    if patience_counter <= 0:
                        tqdm.write(f"\nEarly stopping at epoch {epoch_num}.")
                        break

                epoch_pbar.close()

            # ========== MODEL SAVING ==========
            if best_model_state:
                model.load_state_dict(best_model_state)

                os.makedirs("bin", exist_ok=True)
                save_dict = {
                    "model_state_dict": best_model_state,
                    "bert_model_name": BERT_MODEL_NAME,
                    "dropout_rate": DROPOUT_RATE,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "intents_vocab": intents_vocab,
                    "slots_vocab": slots_vocab,
                    "config_name": config_name,
                    "num_intent_labels": len(lang.intent2id),
                    "num_slot_labels": len(lang.slot2id),
                    "intent2id": lang.intent2id,
                    "slot2id": lang.slot2id,
                }

                model_save_path = os.path.join(
                    "bin",
                    f"{BERT_MODEL_NAME.split('/')[-1]}_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{current_batch_size}.pt",
                )
                torch.save(save_dict, model_save_path)
                print(f"Saved: {model_save_path.split('/')[-1]}")

            # ========== FINAL EVALUATION ==========
            if best_model_state:
                model.load_state_dict(best_model_state)
                test_f1, test_acc, test_loss = eval_loop_bert(
                    test_loader,
                    criterion_slots,
                    criterion_intents,
                    model,
                    lang,
                    tokenizer,
                )

                print(
                    f"Results: F1={test_f1:.4f} | Acc={test_acc:.4f} | Loss={test_loss:.4f}"
                )

                # Store results for comparison
                all_results.append(
                    {
                        "model": BERT_MODEL_NAME,
                        "config": config_name,
                        "test_f1": test_f1,
                        "test_acc": test_acc,
                        "test_loss": test_loss,
                    }
                )

            else:
                print(f"ERROR: No model state available for {BERT_MODEL_NAME}")

        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  OOM with batch size {current_batch_size}, trying smaller batch size..."
            )
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Try smaller batch size
            current_batch_size = 32
            print(f"  Retrying with batch size: {current_batch_size}")

            try:
                # Recreate data loaders with smaller batch size
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=current_batch_size,
                    collate_fn=collate_fn_bert,
                    shuffle=True,
                )
                dev_loader = (
                    DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn_bert)
                    if dev_dataset
                    else None
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=32, collate_fn=collate_fn_bert
                )

                # Recreate model
                model = JointBERTModel(
                    num_intent_labels=len(lang.intent2id),
                    num_slot_labels=len(lang.slot2id),
                    bert_model_name=BERT_MODEL_NAME,
                    dropout_rate=DROPOUT_RATE,
                ).to(device)

                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

                # Training with smaller batch size
                best_f1_dev = 0.0
                patience_counter = EARLY_STOP_PATIENCE
                best_model_state = copy.deepcopy(model.state_dict())

                if not dev_loader:
                    for epoch in tqdm(
                        range(1, MAX_EPOCHS + 1),
                        desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]} (reduced BS)",
                        unit="epoch",
                    ):
                        train_loop_bert(
                            train_loader,
                            optimizer,
                            criterion_slots,
                            criterion_intents,
                            model,
                            CLIP,
                        )
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    epoch_pbar = tqdm(
                        range(1, MAX_EPOCHS + 1),
                        desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]} (BS={current_batch_size})",
                        unit="epoch",
                    )

                    for epoch_num in epoch_pbar:
                        train_loss = train_loop_bert(
                            train_loader,
                            optimizer,
                            criterion_slots,
                            criterion_intents,
                            model,
                            CLIP,
                        )
                        dev_f1, dev_acc, dev_loss = eval_loop_bert(
                            dev_loader,
                            criterion_slots,
                            criterion_intents,
                            model,
                            lang,
                            tokenizer,
                        )
                        epoch_pbar.set_postfix(
                            {
                                "TrainLoss": f"{train_loss:.4f}",
                                "DevLoss": f"{dev_loss:.4f}",
                                "DevF1": f"{dev_f1:.4f}",
                                "DevAcc": f"{dev_acc:.4f}",
                            }
                        )
                        if dev_f1 > best_f1_dev:
                            best_f1_dev = dev_f1
                            best_model_state = copy.deepcopy(model.state_dict())
                            patience_counter = EARLY_STOP_PATIENCE
                        else:
                            patience_counter -= 1
                        if patience_counter <= 0:
                            tqdm.write(f"\nEarly stopping at epoch {epoch_num}.")
                            break
                    epoch_pbar.close()

                # Save and evaluate with reduced batch size
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    os.makedirs("bin", exist_ok=True)
                    save_dict = {
                        "model_state_dict": best_model_state,
                        "bert_model_name": BERT_MODEL_NAME,
                        "dropout_rate": DROPOUT_RATE,
                        "max_seq_length": MAX_SEQ_LENGTH,
                        "intents_vocab": intents_vocab,
                        "slots_vocab": slots_vocab,
                        "config_name": config_name,
                        "num_intent_labels": len(lang.intent2id),
                        "num_slot_labels": len(lang.slot2id),
                        "intent2id": lang.intent2id,
                        "slot2id": lang.slot2id,
                    }
                    model_save_path = os.path.join(
                        "bin",
                        f"{BERT_MODEL_NAME.split('/')[-1]}_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{current_batch_size}.pt",
                    )
                    torch.save(save_dict, model_save_path)
                    print(f"Saved: {model_save_path.split('/')[-1]}")

                    test_f1, test_acc, test_loss = eval_loop_bert(
                        test_loader,
                        criterion_slots,
                        criterion_intents,
                        model,
                        lang,
                        tokenizer,
                    )
                    print(
                        f"Results: F1={test_f1:.4f} | Acc={test_acc:.4f} | Loss={test_loss:.4f}"
                    )
                    all_results.append(
                        {
                            "model": BERT_MODEL_NAME,
                            "config": config_name,
                            "test_f1": test_f1,
                            "test_acc": test_acc,
                            "test_loss": test_loss,
                        }
                    )

            except torch.cuda.OutOfMemoryError:
                print(f"  SKIPPED: Still OOM even with batch size {current_batch_size}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            # Clear GPU cache and continue with next model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # ========== FINAL COMPARISON ==========
    print(f"\nFinal Comparison:")

    if all_results:
        print(f"{'Model':<20} {'F1':<8} {'Acc':<8} {'Loss':<8}")
        print("-" * 45)
        for result in all_results:
            model_short = result["model"].split("-")[-1]  # Get 'uncased' part
            print(
                f"bert-{model_short:<15} {result['test_f1']:<8.4f} {result['test_acc']:<8.4f} {result['test_loss']:<8.4f}"
            )

        # Find best model
        best_model = max(all_results, key=lambda x: x["test_f1"])
        print(f"\nBest: {best_model['model']} (F1: {best_model['test_f1']:.4f})")

    print(f"\nTraining completed.")
