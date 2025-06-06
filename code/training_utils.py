from tqdm import trange
import torch
import torch.nn as nn
import os
from .data_prep import *
from .model import *
from .metrics import *
from .result_eval import *
from torch.utils.data import DataLoader
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Loss + Save
# Loss function that handles both predicted feature vectors and IPA tokens
def loss_function(featvec_preds, char_preds, featvec_target, char_target, vocab_size):
    # Feature vector loss
    featvec_criterion = nn.CrossEntropyLoss(ignore_index = 4)
    featvec_preds = featvec_preds.view(-1, 5)
    featvec_target = featvec_target.view(-1).to(torch.long)
    featvec_loss = featvec_criterion(featvec_preds, featvec_target)

    # IPA token loss
    char_loss = nn.CrossEntropyLoss(ignore_index=0)
    char_preds = char_preds.view(-1, vocab_size)
    char_target = char_target.view(-1)
    char_loss = char_loss(char_preds, char_target)

    return featvec_loss + char_loss

def save_model(model, name = "model"):
    # Check if model directory exists
    if not os.path.exists("./model_checkpoints"):
        os.makedirs("./model_checkpoints/")
    pth = os.path.join("./model_checkpoints", f"{name}.pt")
    torch.save(model.state_dict(), pth)

## Reflex prediction
def training(model, train_dataloader, dev_dataloader, optimizer, n_epochs, n_steps, save_name, save = True, print_loss = False):
    eval_loss = 0
    for epoch in trange(n_epochs, desc="Epoch", position=0):
        model.train()
        total_loss = 0
        for step, (langs, valids, target_langs, target, target_featvec) in enumerate(train_dataloader):
            langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)

            optimizer.zero_grad()
            feat_logits, ipa_logits = model(langs, valids, target_langs, target, target_featvec)
            loss = loss_function(feat_logits, ipa_logits, target_featvec, target, model.vocab_size)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step == n_steps:
                break
        if print_loss:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        model.eval()
        epoch_eval_loss = 0
        with torch.no_grad():
            for langs, valids, target_langs, target, target_featvec in dev_dataloader:
                langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)
                feat_logits, ipa_logits = model(langs, valids, target_langs, target, target_featvec)
                loss = loss_function(feat_logits, ipa_logits, target_featvec, target, model.vocab_size)
                epoch_eval_loss += loss.item()
        eval_loss = epoch_eval_loss / len(dev_dataloader)
        if print_loss:
            print(f"Validation Loss: {eval_loss:.4f}")
    # Save model at the end of training
    if save:
        save_model(model, save_name)
    return model

def reflex_finetune(args, model_name, filename, missing_prop, 
                    fine_tune_epochs = 8, fine_tune_steps = 100,
                    fine_tune_batch_size = 48, fine_tune_lr = 1e-4):
    data = load_reflex_data(filename)
    train = f'train-{missing_prop}'
    test = f'test-{missing_prop}'

    data['dev'] = {'data':[]}
    data[train]['data'], data['dev']['data'] = train_test_split(data[train]['data'], test_size= 0.01)
    train_data = data[train]
    test_data = data[test]

    # Load vocab
    with open(f"./model_checkpoints/{model_name}_vocab.pkl", "rb") as f:
        char2featvec, char2idx = pickle.load(f)
    idx2char = {v: k for k, v in char2idx.items()}
    
    # Prep data
    train_dataset = TrainDataset(train_data, char2featvec, char2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=fine_tune_batch_size, collate_fn=collate_fn, drop_last=True)
    dev_dataset = DevDataset(data['dev'], char2featvec, char2idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size=fine_tune_batch_size, collate_fn=collate_fn)
    test_dataset = TestDataset(test_data, char2featvec, char2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=fine_tune_batch_size, collate_fn=collate_fn)

    # Load model
    model = CognateS2S(
        char2idx,
        feat_dim=args["feat_vec_dim"],
        hidden_dim=args["hidden_dim"],
        num_heads=args["nhead"],
        num_encoder_layers=args["nlayers_enc"],
        num_decoder_layers=args["nlayers_dec"],
        dropout=args["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(f"./model_checkpoints/{model_name}.pt"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr)
    model = training(model, 
                    train_dataloader, 
                    dev_dataloader, 
                    optimizer, 
                    fine_tune_epochs,
                    fine_tune_steps,
                    model_name,
                    save = False,
                    print_loss = args["print_loss"])

    results = compute_metrics(model, test_dataloader, char2idx, idx2char)

    # Export predictions
    export_predictions_reflex(model, test_dataloader, char2idx, filename.split('/')[-1], model_name, missing_prop, device)
    return results


## Proto reconstruction

def pretrain_proto(data, model_name, args):
    pretrain_data = data['pretrain']
    pretrain_data['dev'] = {'data':[]}
    pretrain_data['train'] = {'data':[]}
    pretrain_data['train']['data'], pretrain_data['dev']['data'] = train_test_split(pretrain_data['data'], test_size= 0.08)
    train_data = pretrain_data['train']
    dev_data = pretrain_data['dev']
    vocab, languages = get_vocab(train_data)
    # Add languages and vocab from finetuning data 
    vocab2, languages2 = get_vocab(data['finetune_train_0.1_1'], key = 'solns')
    vocab = list(set(vocab + vocab2))   
    languages = list(set(languages + languages2))
    char2featvec, char2idx = tokenizer(vocab, languages)


    train_dataset = TrainDataset(train_data, char2featvec, char2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, drop_last=True)
    dev_dataset = DevDataset(dev_data, char2featvec, char2idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, shuffle=False)


    # Initialize model
    model = CognateS2S(
        char2idx,
        feat_dim=args["feat_vec_dim"],
        hidden_dim=args["hidden_dim"],
        num_heads=args["nhead"],
        num_encoder_layers=args["nlayers_enc"],
        num_decoder_layers=args["nlayers_dec"],
        dropout=args["dropout"]
    ).to(device)


    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])
    model = training(model, 
                    train_dataloader, 
                    dev_dataloader, 
                    optimizer, 
                    args["n_epochs"],
                    args["n_steps"],
                    model_name,
                    save = True,
                    print_loss = args["print_loss"])
    with open(f"./model_checkpoints/{model_name}_vocab.pkl", "wb") as f:
        pickle.dump((char2featvec, char2idx), f)

def training_proto_finetune(model, train_dataloader, dev_dataloader, optimizer, n_epochs, save_name, save = True, print_loss = False):
    eval_loss = 0
    for epoch in trange(n_epochs, desc="Epoch", position=0):
        model.train()
        total_loss = 0
        for langs, valids, target_langs, target, target_featvec in train_dataloader:
            langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)

            optimizer.zero_grad()
            feat_logits, ipa_logits = model(langs, valids, target_langs, target, target_featvec)
            loss = loss_function(feat_logits, ipa_logits, target_featvec, target, model.vocab_size)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        epoch_eval_loss = 0
        with torch.no_grad():
            for langs, valids, target_langs, target, target_featvec in dev_dataloader:
                langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)
                feat_logits, ipa_logits = model(langs, valids, target_langs, target, target_featvec)
                loss = loss_function(feat_logits, ipa_logits, target_featvec, target, model.vocab_size)
                epoch_eval_loss += loss.item()
        eval_loss = epoch_eval_loss / len(dev_dataloader)
        if print_loss:
            print(f"Validation Loss: {eval_loss:.4f}")
    # Save model at the end of training
    if save:
        save_model(model, save_name)
    return model

def proto_finetune(model_name, args, missing_prop = '0.1', 
              fold = 1, fine_tune_epochs = 20, fine_tune_lr = 5e-4,
              fine_tune_bs = 48, data=None, filename=None, proto_map=None, family=None):
    if data is None:
        data = load_proto_data(proto_map, filename)
    # Load fine-tuning data
    train_data = data[f'finetune_train_{missing_prop}_{fold}']
    dev_data = data[f'finetune_dev_{missing_prop}_{fold}']
    test_data = data[f'finetune_test_{missing_prop}_{fold}']

    # Load vocab
    with open(f"./model_checkpoints/{model_name}_vocab.pkl", "rb") as f:
        char2featvec, char2idx = pickle.load(f)
    idx2char = {v: k for k, v in char2idx.items()}

    train_dataset = ProtoDataset(train_data, char2featvec, char2idx)
    train_dataloader = DataLoader(train_dataset, batch_size=fine_tune_bs, collate_fn=collate_fn, shuffle = True)
    dev_dataset = ProtoDataset(dev_data, char2featvec, char2idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size=fine_tune_bs, collate_fn=collate_fn)
    test_dataset = ProtoDataset(test_data, char2featvec, char2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=fine_tune_bs, collate_fn=collate_fn)

    model = CognateS2S(
        char2idx,
        feat_dim=args["feat_vec_dim"],
        hidden_dim=args["hidden_dim"],
        num_heads=args["nhead"],
        num_encoder_layers=args["nlayers_enc"],
        num_decoder_layers=args["nlayers_dec"],
        dropout=args["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(f"./model_checkpoints/{model_name}.pt"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr)

    model = training_proto_finetune(model, 
                    train_dataloader, 
                    dev_dataloader, 
                    optimizer, 
                    fine_tune_epochs,
                    model_name, 
                    save = False,
                    print_loss = args["print_loss"])

    results = compute_metrics(model, test_dataloader, char2idx, idx2char)
    # Export predictions
    if family is not None:
        export_predictions_proto(model, test_dataloader, char2idx, family, model_name, missing_prop, fold, device)
    return results
