import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm

from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_VarDropWT  


if __name__ == "__main__":

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1B_Step3"  
    model_save_dir = "bin"  

    # Hyperparameters - these control the model architecture and training behavior
    lr = 3.75  # Learning rate
    batch_size = 16
    hid = 300  # Hidden size
    emb = 300  # Embedding size (must equal hidden size for weight tying)
    clip = 5   # Gradient clipping value
    n_epochs = 100
    patience = 7  # Early stopping patience
    nt_avsgd_trigger = 3  # Epochs to wait before triggering NT-AvSGD
    emb_dropout = 0.1  # Embedding dropout rate
    fc_dropout = 0.4   # Output layer dropout rate

    # Load Penn Tree Bank dataset and create vocabulary
    print("\nLoading data & vocabulary...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang = Lang(train_raw, ["<pad>", "<eos>"])  # Create word-to-id mapping
        vocab_len = len(lang.word2id)
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"Error: Dataset not found: {e}")
        exit()

    # Create PyTorch datasets for training, validation, and testing
    print("Creating Datasets...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")

    print(f"\n--- Running Configuration: {experiment_step} ---")
    print(f"   LR={lr}, BS={batch_size}, Optim=NT-AvSGD(SGD)")
    print(f"   Model=LM_LSTM_VarDropWT, Patience={patience}, NT-AvSGD Trigger={nt_avsgd_trigger}")

    # Create data loaders with custom collate function for padding sequences
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE),
        shuffle=True,
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,  # Larger batch size for evaluation (no gradients needed)
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE)
    )

    # Initialize model with variational dropout and weight tying
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_VarDropWT(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=lang.word2id["<pad>"],
        n_layers=1,
        dropoute=emb_dropout,  # Variational dropout on embeddings
        dropouto=fc_dropout    # Variational dropout on output
    ).to(DEVICE)
    model.apply(init_weights)  # Apply custom weight initialization

    # Setup optimizer and loss functions
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Training state variables
    print("Starting Training...")
    patience_counter = patience
    trigger_counter = nt_avsgd_trigger
    best_ppl = math.inf
    best_model = None

    # NT-AvSGD (Non-monotonic Triggered Averaging SGD) variables
    # This technique averages model parameters over time to improve convergence
    averaging_active = False      # Whether parameter averaging is active
    averaged_params = None        # Stores the running average of parameters
    steps_since_averaging = 0     # Counter for averaging steps
    trigger_best_ppl = math.inf   # Best perplexity before triggering averaging
    
    # For tracking training progress
    sampled_epochs = []
    losses_train = []
    losses_dev = []

    # Main training loop
    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        # Train for one epoch
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
 
        # NT-AvSGD: Update parameter averaging if active
        if averaging_active:
            steps_since_averaging += 1
            if averaged_params is None:
                # Initialize averaged parameters on first step
                averaged_params = {}
                for name, param in model.named_parameters():
                    averaged_params[name] = param.data.clone()
            else:
                # Update running average: avg = avg + (new - avg) / n
                for name, param in model.named_parameters():
                    averaged_params[name] = averaged_params[name] + \
                        (param.data - averaged_params[name]) / steps_since_averaging
        
        # Evaluate model performance every epoch
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss)
            
            # If averaging is active, evaluate using averaged parameters
            if averaging_active and averaged_params is not None:
                # Temporarily replace model parameters with averaged ones
                saved_params = {}
                for name, param in model.named_parameters():
                    saved_params[name] = param.data.clone()
                
                for name, param in model.named_parameters():
                    param.data = averaged_params[name].to(DEVICE)
                
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                
                # Restore original parameters
                for name, param in model.named_parameters():
                    param.data = saved_params[name]
            else:
                # Standard evaluation
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            
            losses_dev.append(loss_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            
            # NT-AvSGD trigger logic: activate averaging if no improvement for several epochs
            if not averaging_active:
                if ppl_dev < trigger_best_ppl:
                    trigger_best_ppl = ppl_dev
                    trigger_counter = nt_avsgd_trigger
                else:
                    trigger_counter -= 1
                    if trigger_counter <= 0:
                        print(f"\nNT-AvSGD triggered at epoch {epoch}!")
                        averaging_active = True
                        steps_since_averaging = 0
            
            # Save best model and implement early stopping
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                # Save model with averaged parameters if averaging is active
                if averaging_active and averaged_params is not None:
                    best_model = copy.deepcopy(model).to('cpu')
                    for name, param in best_model.named_parameters():
                        param.data = averaged_params[name].to('cpu')
                else:
                    best_model = copy.deepcopy(model).to('cpu')
                patience_counter = patience
            else:
                patience_counter -= 1
                
            if patience_counter <= 0:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Final evaluation on test set using the best model
    if best_model is not None:
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)

        # Model saving code (commented out)
        #print("\nSaving best model...")
        #os.makedirs(model_save_dir, exist_ok=True)
        #model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}.pt"
        #model_save_path = os.path.join(model_save_dir, model_filename)
        #try:
            #torch.save(best_model.state_dict(), model_save_path)
            #print(f"Model saved to: {model_save_path}")
        #except Exception as e:
            #print(f"Error saving model: {e}")
    else:
        print("\nNo best model found.")

    print(f"\n--- Script {experiment_step} Finished ---")