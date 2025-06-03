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

    
    lr = 3.75  
    batch_size = 16
    hid = 300
    emb = 300  
    clip = 5
    n_epochs = 100
    patience = 7
    nt_avsgd_trigger = 3  
    emb_dropout = 0.1  
    fc_dropout = 0.4  


    print("\nLoading data & vocabulary...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang = Lang(train_raw, ["<pad>", "<eos>"])
        vocab_len = len(lang.word2id)
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"Error: Dataset not found: {e}")
        exit()

    print("Creating Datasets...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")


    print(f"\n--- Running Configuration: {experiment_step} ---")
    print(f"   LR={lr}, BS={batch_size}, Optim=NT-AvSGD(SGD)")
    print(f"   Model=LM_LSTM_VarDropWT, Patience={patience}, NT-AvSGD Trigger={nt_avsgd_trigger}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE),
        shuffle=True,
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,  
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  
        collate_fn=partial(collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE)
    )


    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_VarDropWT(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=lang.word2id["<pad>"],
        n_layers=1,
        dropoute=emb_dropout,
        dropouto=fc_dropout
    ).to(DEVICE)
    model.apply(init_weights) 

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    print("Starting Training...")
    patience_counter = patience
    trigger_counter = nt_avsgd_trigger
    best_ppl = math.inf
    best_model = None

    averaging_active = False
    averaged_params = None
    steps_since_averaging = 0
    trigger_best_ppl = math.inf
    
    sampled_epochs = []
    losses_train = []
    losses_dev = []

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
 
        if averaging_active:
            steps_since_averaging += 1
            if averaged_params is None:
    
                averaged_params = {}
                for name, param in model.named_parameters():
                    averaged_params[name] = param.data.clone()
            else:
        
                for name, param in model.named_parameters():
                    averaged_params[name] = averaged_params[name] + \
                        (param.data - averaged_params[name]) / steps_since_averaging
        
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss)
            

            if averaging_active and averaged_params is not None:

                saved_params = {}
                for name, param in model.named_parameters():
                    saved_params[name] = param.data.clone()
                
                for name, param in model.named_parameters():
                    param.data = averaged_params[name].to(DEVICE)
                
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                

                for name, param in model.named_parameters():
                    param.data = saved_params[name]
            else:
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            
            losses_dev.append(loss_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            

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
            

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
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

    if best_model is not None:
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)

        # Save the best model
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