import numpy as np
import os
from lingrex.reconstruct import eval_by_bcubes, eval_by_dist
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_to_string(seqs, char2idx, idx2char):
    decoded_seq = []
    for i in range(len(seqs)):
        seqs_str = []
        for char in seqs[i]:
            if char == char2idx["[SEP]"]:
                break
            seqs_str.append(idx2char.get(char, "[UNK]"))
        decoded_seq.append(" ".join(seqs_str))
    return decoded_seq


def compute_metrics(model, data, char2idx, idx2char):
    model.eval()
    decoded_preds = []
    decoded_labels = []
    target_langs = []
    for batch in tqdm(data, desc="Computing metrics"):
        with torch.no_grad():
            langs, valids, masked_langs, masked_value, _ = batch
            langs, valids, masked_langs = langs.to(device), valids.to(device), masked_langs.to(device)
            preds = model.generate(langs, valids, masked_langs)
            preds = preds.cpu().numpy()
            masked_value = masked_value.cpu().numpy()
            masked_langs = masked_langs.cpu().numpy()
            decoded_preds += translate_to_string(preds, char2idx, idx2char)
            decoded_labels += translate_to_string(masked_value, char2idx, idx2char)
            target_langs += translate_to_string(masked_langs, char2idx, idx2char)

    
    languages = {}
    
    for x,y,l in zip(decoded_preds, decoded_labels, target_langs):
        lng = l
        if lng not in languages:
            languages[lng] = {'preds':[], 'labels':[]}
        languages[lng]['preds'].append(x) 
        languages[lng]['labels'].append(y) 

    
    result = np.array([0.0, 0.0, 0.0])
    for lng, D in languages.items():
        p = np.array(D['preds'])
        l = np.array(D['labels'])
       
        languages[lng]['res'] = np.array([ eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)]), \
                               eval_by_dist([[x.split(), y.split()] for x,y in zip(p,l)], normalized= True), \
                               eval_by_bcubes([[x.split(), y.split()] for x,y in zip(p,l)])])
        result += languages[lng]['res']

    result /= len(languages)
    result = {'Avg ED': result[0],
              'Avg NED': result[1],
              'B^3 F1' : result[2]}
    
    result = {key: round(value,4) for key, value in result.items()}
    result.update({lng: '\t'.join([str(x) for x in languages[lng]['res'].round(4)]) for lng in languages})

    return result


# Average metrics over language family
def metrics_by_fam(eval_dict):
    path = "./data/reflex_data/data-surprise/"
    dirs = os.listdir(path)

    families = {}
    for d in dirs:
        if os.path.isdir(path + d):
            files = os.listdir(path + d)
            for f in files:
                if 'cognates' in f:
                    with open(path + d + '/' + f, encoding="UTF-8") as file:
                        header = file.readline().strip().split('\t')[1:]
                        header = ['[' + x + ']' for x in header]
                        break
            families[d] = header

    fam_results = {}
    for fam, langs in families.items():
        fam_results[fam] = {'ED':[], 'NED':[], 'B3':[]}
        for lang in langs:
            if lang in eval_dict:
                ed,ned,b3 = eval_dict[lang].split('\t')
                fam_results[fam]['ED'].append(float(ed))
                fam_results[fam]['NED'].append(float(ned))
                fam_results[fam]['B3'].append(float(b3))
        fam_results[fam]['ED'] = np.round(np.mean(fam_results[fam]['ED']), decimals=4)
        fam_results[fam]['NED'] = np.round(np.mean(fam_results[fam]['NED']), decimals=4)
        fam_results[fam]['B3'] = np.round(np.mean(fam_results[fam]['B3']), decimals=4)
    return fam_results

def print_examples(model, dataloader, all2id, all2id_inv, n_examples=5):
    model.eval()
    to_print = 0

    for langs, valids, target_langs, target, target_featvec in dataloader:
        langs, valids, target_langs, target, target_featvec = langs.to(device), valids.to(device), target_langs.to(device), target.to(device), target_featvec.to(device)
        if to_print == n_examples:
                break
        with torch.no_grad():
            predictions = model.generate(langs, valids, target_langs)

        # Convert predictions to strings
        predictions = predictions.cpu().numpy()
        target = target.cpu().numpy()
        predictions_str = []
        target_str = []
        for i in range(len(predictions)):
            pred_str = []
            target_str.append(" ".join([all2id_inv[char] for char in target[i] if char != 0][:-1]))
            for char in predictions[i]:
                if char == all2id["[SEP]"]:
                    break
                pred_str.append(all2id_inv[char])
            predictions_str.append(" ".join(pred_str))

        for target, pred in zip(target_str, predictions_str):
            print(f"Target: {target}")
            print(f"Prediction: {pred}")
            print()
            to_print += 1
            if to_print == n_examples:
                break