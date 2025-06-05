import pandas as pd
import os
import torch
from tqdm import tqdm
import json


def avg_results_reflex(results_paths, save_name = "reflex_finetuned_avg.txt"):
    res = []
    for results_path in results_paths:
        df = pd.read_csv(results_path, sep='\t')
        avg_ed = round(df['ED'].mean(), 4)
        avg_ned = round(df['NED'].mean(), 4)
        avg_b3 = round(df['B3_F1'].mean(), 4)
        if "0.50" in results_path:
            prop = "0.50"
        elif "0.30" in results_path:
            prop = "0.30"
        else:
            prop = "0.10"
        print("Reflex prediction results (averaged over languages):")
        print(f'Prop: {prop} || Average ED: {avg_ed}')
        print(f'Prop: {prop} || Average NED: {avg_ned}')
        print(f'Prop: {prop} || Average B3: {avg_b3}')
        print('\n')
        res.append([prop, avg_ed, avg_ned, avg_b3])
    # Write to file
    with open('./results/' + save_name, 'w') as f:
        f.write('Proportion\tED\tNED\tB3_F1\n')
        for r in res:
            f.write(f'{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n')


def avg_results_proto(results_path, save_name = "proto_avg.txt"):
    df = pd.read_csv(results_path, sep='\t')
    # Group by Proportion
    grouped = df.groupby('Proportion')
    avg_results = []
    for prop, group in grouped:
        avg_ed = round(group['ED'].mean(), 4)
        avg_ned = round(group['NED'].mean(), 4)
        avg_b3 = round(group['B3_F1'].mean(), 4)
        print("Proto-language reconstruction results (averaged over languages and folds):")
        print(f'Prop: {prop} || Average ED: {avg_ed}')
        print(f'Prop: {prop} || Average NED: {avg_ned}')
        print(f'Prop: {prop} || Average B3: {avg_b3}')
        print('\n')
        avg_results.append([prop, avg_ed, avg_ned, avg_b3])
    # Write to file
    with open('./results/' + save_name, 'w') as f:
        f.write('Proportion\tED\tNED\tB3_F1\n')
        for res in avg_results:
            f.write(f'{res[0]}\t{res[1]}\t{res[2]}\t{res[3]}\n')

### Export predictions
def export_predictions_reflex(model, dataloader, char2idx, family, model_name, missing_prop, device):
    idx2char = {v: k for k, v in char2idx.items()}
    model.eval()
    predictions = []
    with torch.no_grad():
        for langs, valids, target_langs, target, target_featvec in tqdm(dataloader, desc="Exporting predictions"):
            langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)
            preds = model.generate(langs, valids, target_langs)
            preds = preds.cpu().numpy()
            for i in range(len(preds)):
                pred_str = []
                for char in preds[i]:
                    if char == char2idx["[SEP]"]:
                        break
                    pred_str.append(idx2char[char])
                predictions.append(" ".join(pred_str))
    # Load solution file
    sol_df = pd.read_csv(f"./data/reflex_data/data-surprise/{family}/solutions-{missing_prop}.tsv", sep="\t")
    sol_df = sol_df.melt(id_vars=["COGID"], var_name="LANGUAGE", value_name="GOLD")
    sol_df = sol_df.dropna(subset=["GOLD"])
    sol_df["PREDICTION"] = predictions
    # Save to tsv
    if not os.path.exists(f"./results/predictions/reflex/{family}"):
        os.makedirs(f"./results/predictions/reflex/{family}")
    sol_df.to_csv(f"./results/predictions/reflex/{family}/{model_name}_predictions.tsv", sep="\t", index=False)


def export_predictions_proto(model, dataloader, all2id, family, model_name, missing_prop, fold, device):
    all2id_inv = {v: k for k, v in all2id.items()}
    model.eval()
    predictions = []
    with torch.no_grad():
        for langs, valids, target_langs, target, target_featvec in tqdm(dataloader, desc="Exporting predictions"):
            langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)
            preds = model.generate(langs, valids, target_langs)
            preds = preds.cpu().numpy()
            for i in range(len(preds)):
                pred_str = []
                for char in preds[i]:
                    if char == all2id["[SEP]"]:
                        break
                    pred_str.append(all2id_inv[char])
                predictions.append(" ".join(pred_str))
    # Load solution file
    sol_pth = f"./data/proto_data/data-{missing_prop}/testitems/{family}/test-{fold}.json"
    f_json = json.load(open(sol_pth,'r'))
    sols = []
    for line in f_json:
        sols.append(' '.join(line[1]))

    # Create a DataFrame with the predictions and solutions
    sol_df = pd.DataFrame({
        'gold': sols,
        'predictions': predictions
    })
    # Save to tsv
    if not os.path.exists(f"./results/predictions/proto/{family}"):
        os.makedirs(f"./results/predictions/proto/{family}")
    sol_df.to_csv(f"./results/predictions/proto/{family}/{model_name}_{missing_prop}_{fold}_predictions.tsv", sep="\t", index=False)
