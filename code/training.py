import torch
from .data_prep import *
from .tokenizer import *
from .metrics import *
from .model import *
from .training_utils import *
from .result_eval import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_reflex(args, language_families):
    data = load_reflex_data()
    missing_props = ['0.50', '0.30', '0.10'] if args["missing_prop"] is None else args["missing_prop"]

    for missing_prop in missing_props:
        lines_language = [f"Family\tLanguage\tED\tNED\tB3_F1"]
        lines_family = [f"Family\tED\tNED\tB3_F1"]
        model_name = "s2s_reflex_" + args["model_suffix"] + missing_prop 
        train = f'train-{missing_prop}'
        test = f'test-{missing_prop}'

        # Split data into train and test
        data['dev'] = {'data':[]}
        data[train]['data'], data['dev']['data'] = train_test_split(data[train]['data'], test_size= 0.08)
        train_data = data[train]
        test_data = data[test]
        vocab, languages = get_vocab(train_data)
        char2featvec, char2idx = tokenizer(vocab, languages)
        idx2char = {v: k for k, v in char2idx.items()}

        # Create dataloaders
        train_dataset = TrainDataset(train_data, char2featvec, char2idx)
        train_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, drop_last=True)
        dev_dataset = DevDataset(data['dev'], char2featvec, char2idx)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, shuffle=False)
        test_dataset = TestDataset(test_data, char2featvec, char2idx)
        test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], collate_fn=collate_fn, shuffle=False)

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
        if args["pretrain"]:
            print(f"(Pre-)Training model with {missing_prop} missing data.")
            model = training(model, 
                            train_dataloader, 
                            dev_dataloader, 
                            optimizer, 
                            args["n_epochs"],
                            args["num_steps"],
                            model_name,
                            save = True)
            with open(f"./model_checkpoints/{model_name}_vocab.pkl", "wb") as f:
                pickle.dump((char2featvec, char2idx), f)
        else:
            if not os.path.exists(f"./model_checkpoints/{model_name}.pt"):
                print(f"Model {model_name} does not exist. Please set PRETRAIN to True.")
                continue
        
        if args["fine_tune"]:
            # Fine-tune on individual datasets
            for family in tqdm(language_families):
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"~~~~~ Fine-tuning on {family} ~~~~~")
                print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                filename = "./data/reflex_data/data-surprise/" + family
                results = reflex_finetune(args, model_name, filename, missing_prop, 
                                    fine_tune_epochs=args["fine_tune_epochs"], fine_tune_steps=args["fine_tune_steps"],
                                    fine_tune_batch_size=args["fine_tune_batch_size"], fine_tune_lr=args["fine_tune_lr"])
                lines_family.append(f"{family}\t{results['Avg ED']}\t{results['Avg NED']}\t{results['B^3 F1']}")
                for key in results:
                    if key != "Avg ED" and key != "Avg NED" and key != "B^3 F1":
                        lines_language.append(f"{family}\t{key}\t{results[key]}")
            family_results = "\n".join(lines_family)
            language_results = "\n".join(lines_language)
            with open(f"./results/{model_name}_language_results_finetuned.txt", "w") as f:
                f.write(language_results)
            with open(f"./results/{model_name}_family_results_finetuned.txt", "w") as f:
                f.write(family_results)
            print(f"Results for {model_name} saved (finetuned).")
        
    
        else:
            results = compute_metrics(model, test_dataloader, char2idx, idx2char)
            results_by_fam = metrics_by_fam(results)
            for key in results_by_fam:
                lines_family.append(f"{key}\t{results_by_fam[key]['ED']}\t{results_by_fam[key]['NED']}\t{results_by_fam[key]['B3']}")
            lines_language = [f"Language\tED\tNED\tB3_F1"]
            for key in results:
                    if key != "Avg ED" and key != "Avg NED" and key != "B^3 F1":
                        lines_language.append(f"{key}\t{results[key]}")
            family_results = "\n".join(lines_family)
            language_results = "\n".join(lines_language)
            with open(f"./results/{model_name}_language_results.txt", "w") as f:
                f.write(language_results)
            with open(f"./results/{model_name}_family_results.txt", "w") as f:
                f.write(family_results)
            print(f"Results for {model_name} saved (no finetuning).")
    # Aggregate results
    if args["fine_tune"]:
        results_paths = [f'./results/{path}' for path in os.listdir('./results') if 'language' in path and 'finetuned' in path and args["model_suffix"] in path and 'proto' not in path]
        outname = "s2s_reflex_" + args["model_suffix"] + "_avg_finetuned.txt"
        avg_results_reflex(results_paths, save_name=outname)
    else:
        results_paths = [f'./results/{path}' for path in os.listdir('./results') if 'language' in path and 'finetuned' not in path and args["model_suffix"] in path and 'proto' not in path] 
        outname = "s2s_reflex_" + args["model_suffix"] + "_avg.txt"
        avg_results_reflex(results_paths, save_name=outname)

def train_proto(args, proto_map):
    language_families = list(proto_map.keys())
    data = load_proto_data(proto_map=proto_map, load_pretrain_data=args["use_pretrain_data"])
    model_name = "s2s_proto" + args["model_suffix"] 
    model_name += "_no_pretrain" if not args["use_pretrain_data"] else ""

    # Pre-train model
    if os.path.exists(f"./model_checkpoints/{model_name}.pt") and not args["force_pretrain"]:
        print("Model already trained and pre-training not enforced, skipping pre-training")
    else:
        pretrain_proto(data, model_name, args)


    # Finetuning loop
    if not args["zero_shot"]:
        lines_detailed = [f"Proportion\tFold\tFamily\tED\tNED\tB3_F1"]
        lines_family = [f"Proportion\tFamily\tED\tNED\tB3_F1"]
        missing_props = ['0.8', '0.5', '0.1'] if args["missing_prop"] is None else args["missing_prop"]
        for missing_prop in missing_props: 
            ft_epchs = args["fine_tune_epochs"]
            ft_lr = args["fine_tune_lr"]
            if missing_prop == '0.8': 
                ft_bs = args["fine_tune_batch_size_80"]
            else:
                ft_bs = args["fine_tune_batch_size"]
            # Fine-tune on test data
            for family in tqdm(language_families, desc="Fine-tuning on families"):
                data = load_proto_data(proto_map, family)
                res = np.array([0.0,0.0,0.0])
                for fold in range(1, args["n_folds"]+1): 
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"~~~~~ Fine-tuning on {family} ~~~~~")
                    print(f"~~~~~ Missing Proportion: {missing_prop} ~~~~~")
                    print(f"~~~~~ Fold {fold} / {args['n_folds']} ~~~~~")
                    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    results = proto_finetune(model_name, args,
                                        missing_prop = missing_prop, fold = fold,
                                        fine_tune_epochs = ft_epchs,
                                        fine_tune_lr = ft_lr,
                                        fine_tune_bs = ft_bs,
                                        data = data,
                                        family = family)
                    res += np.array([results['Avg ED'], results['Avg NED'], results['B^3 F1']])
                    lines_detailed.append(f"{missing_prop}\t{fold}\t{family}\t{results['Avg ED']}\t{results['Avg NED']}\t{results['B^3 F1']}")
                res /= args["n_folds"]
                lines_family.append(f"{missing_prop}\t{family}\t{res[0]}\t{res[1]}\t{res[2]}")
                
        family_results = "\n".join(lines_family)
        detailed_results = "\n".join(lines_detailed)
        with open(f"results/{model_name}_detailed_results.txt", "w") as f:
            f.write(detailed_results)
        with open(f"results/{model_name}_family_results.txt", "w") as f:
            f.write(family_results)
        print(f"Results for {model_name} saved.")
        # Aggregate results
        results_path = f'./results/{model_name}_family_results.txt'
        avg_results_proto(results_path, save_name=f"{model_name}_avg.txt")
    
    else: # zero-shot learning
        if not os.path.exists(f"./model_checkpoints/{model_name}.pt"):
            pretrain_proto(data, model_name, args)
        with open(f"./model_checkpoints/{model_name}_vocab.pkl", "rb") as f:
            char2featvec, char2idx = pickle.load(f)
        idx2char = {v: k for k, v in char2idx.items()}
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
        # Proto test data
        lines_detailed = [f"Proportion\tFold\tProto-Lang\tED\tNED\tB3_F1"]
        lines_family = [f"Proportion\tProto-Lang\tED\tNED\tB3_F1"]
        family_results = {'[' + l + ']': np.array([0.0,0.0,0.0]) for f,l in proto_map.items()}
        for fold in range(1, args["n_folds"]+1):
            test_data = data[f'finetune_test_0.8_{fold}']
            test_dataset = ProtoDataset(test_data, char2featvec, char2idx)
            test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)
            results = compute_metrics(model, test_dataloader, char2idx, idx2char)
            for key in results:
                if key != "Avg ED" and key != "Avg NED" and key != "B^3 F1":
                    lines_detailed.append(f"0.8\t{fold}\t{key}\t{results[key]}")
                    family_results[key] += np.array([float(results[key].split('\t')[0]), float(results[key].split('\t')[1]), float(results[key].split('\t')[2])])
        for key in family_results:
            family_results[key] /= args["n_folds"]
            lines_family.append(f"0.8\t{key}\t{family_results[key][0]}\t{family_results[key][1]}\t{family_results[key][2]}")
        detailed_results = "\n".join(lines_detailed)
        family_results = "\n".join(lines_family)

        with open(f"results/{model_name}_zeroshot_detailed_results.txt", "w") as f:
            f.write(detailed_results)
        with open(f"results/{model_name}_zeroshot_family_results.txt", "w") as f:
            f.write(family_results)
        # Aggregate results
        results_path = f'./results/{model_name}_zeroshot_family_results.txt'
        avg_results_proto(results_path, save_name=f"{model_name}_zeroshot_avg.txt")
