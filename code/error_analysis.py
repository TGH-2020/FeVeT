"""
Scripts used for error analysis.
"""


import numpy as np
import torch
from .metrics import translate_to_string
from .model import *
from .training_utils import *
from .data_prep import *
from .tokenizer import *
import lingpy
from itertools import combinations
from soundvectors import FeatureBundle
import pandas as pd
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

featnames = list(FeatureBundle().as_dict().keys())


# Analyze the errors in the predictions
# 1. Most common wrongly predicted characters
# 2. Most common wrongly predicted feature

# Inspired by https://doi.org/10.58079/m6kc
def get_scorer(vocab):
    feat_vecs = {}
    for c in vocab:
        feat_vecs[c] = get_feature_vector(c)
    # Get distance between feature vectors, using a custom distance lookup table
    # Dists from 0 to 1/ 1 to 0 and 2 to 1/ 1 to 2 are 2, all other distances are 1
    # This is because 1 means "not applicable", which indicates a large distance between the two characters
    distmat = np.array([[0, 2, 1], 
                        [2, 0, 2], 
                        [1, 2, 0]])
    scorer = {}
    for a,b in combinations(vocab, r=2):
        dist = 0
        for i in range(len(feat_vecs[a])):
            dist += distmat[feat_vecs[a][i]][feat_vecs[b][i]]
        scorer[a,b] = dist
        scorer[b,a] = dist
        scorer[a,a] = 0
        scorer[b,b] = 0
    
    # Normalize the scores, get similarity instead of distance
    max_score = max(scorer.values())
    for k in scorer:
        scorer[k] = 1-(scorer[k] / max_score)
    return scorer

# Build a position tracker that tracks the position of the characters before and after alignment
# We need this to also align the feature vectors in the exact same way as the characters
#Example:
#pred = ['s', 'e', 'e', 'n', 'e']
#label = ['s', 'e', 'n', 'd']
#alignment = (['s', 'e', 'e', 'n', 'e', '-'], ['s', 'e', '-', 'n', '-', 'd'])
#pred_pos, label_pos = position_tracker(pred, label, alignment)
def position_tracker(pred, label, alignment):
    pred_pos = np.arange(len(pred))
    label_pos = np.arange(len(label))
    for i in range(len(alignment[0])):
        if alignment[0][i] == '-':
            pred_pos = np.insert(pred_pos, i, -1)
    for i in range(len(alignment[1])):
        if alignment[1][i] == '-':
            label_pos = np.insert(label_pos, i, -1)
    return pred_pos, label_pos

def get_alignment(preds, labels):
    vocab = set()
    for x,y in zip(preds, labels):
        vocab.update(x.split())
        vocab.update(y.split())
    vocab = list(vocab)
    scorer = get_scorer(vocab)
    aligned = []
    tracked_positions = []
    for x,y in zip(preds, labels):
        p = x.split()
        l = y.split()
        alignment = lingpy.align.pairwise.nw_align(p, l, scorer=scorer)[0:2]
        aligned.append(alignment)
        pred_pos, label_pos = position_tracker(p, l, alignment)
        tracked_positions.append((pred_pos, label_pos))
    return aligned, tracked_positions


# Takes array of featvecs, returns list where each featvec is cut up to the SEP token
def decode_featvecs(featvecs):
    batch_size = len(featvecs)
    decoded = []
    for i in range(batch_size):
        word = []
        for vec in featvecs[i]:
            if vec[0] == 3:
                break
            word.append(vec)
        decoded.append(word)

    return decoded


def error_analysis(model, data, char2idx, idx2char):
    model.eval()
    decoded_preds = []
    decoded_labels = []
    decoded_preds_fv = []
    decoded_labels_fv = []

    for langs, valids, target_langs, target, target_featvec in data:
        with torch.no_grad():
            langs, valids, target_langs, target = langs.to(device), valids.to(device), target_langs.to(device), target.to(device)
            preds = model.generate(langs, valids, target_langs)
            preds = preds.cpu().numpy()
            target = target.cpu().numpy()
            decoded_preds += translate_to_string(preds, char2idx, idx2char)
            decoded_labels += translate_to_string(target, char2idx, idx2char)

            preds_fv = model.generate(langs, valids, target_langs, output_featvecs=True)
            preds_fv = preds_fv.cpu().numpy()
            target_featvec = target_featvec.to(torch.long).cpu().numpy()
            decoded_preds_fv += decode_featvecs(preds_fv)
            decoded_labels_fv += decode_featvecs(target_featvec)
    
    # Pairwise alignment of the predictions and the labels using LingPy
    # Get all chars from preds and labels into vocab
    aligned, tracked_positions = get_alignment(decoded_preds, decoded_labels)

    errors_ipa = {}
    for x,y in aligned:
        for i in range(len(x)):
            if y[i] != x[i]:
                err_name = y[i] + " / " + x[i]
                if err_name not in errors_ipa:
                    errors_ipa[err_name] = 0
                errors_ipa[err_name] += 1
    # Sort the errors by frequency
    errors_ipa = {k: v for k, v in sorted(errors_ipa.items(), key=lambda item: item[1], reverse=True)}

    ### Now analyze the errors in the feature vectors
    # First, we need to align the feature vectors based on the tracked positions
    aligned_fv = []
    feat_dim = len(decoded_preds_fv[0][0])
    for i in range(len(decoded_preds_fv)):
        p = decoded_preds_fv[i]
        l = decoded_labels_fv[i]
        pred_pos, label_pos = tracked_positions[i]
        # If for some reason there is a mismatch in the lengths of the feature vectors and the characters, we will skip this pair
        if len(p) != max(pred_pos)+1 or len(l) != max(label_pos)+1:
            continue
        p_align = np.full([len(pred_pos), feat_dim], -1)
        l_align = np.full([len(label_pos), feat_dim], -1)
        for j in range(len(pred_pos)):
            if pred_pos[j] != -1:
                p_align[j] = p[pred_pos[j]]
            if label_pos[j] != -1:
                l_align[j] = l[label_pos[j]]
        aligned_fv.append((p_align, l_align))

    # For each pair of preds and labels, compare the two and count the positions where they differ
    # i.e. if the predicted feature vector is [1,2,0,2,1] and the label is [1,2,0,2,2], the error is at position 4
    # In addition to counting the position, we will also note in which direction the error occurs (0 to 1, 0 to 2, 1 to 0, 1 to 2, 2 to 0, 2 to 1)
    errors_fv = {}
    vals = {0.0: '-', 1.0: 'NA', 2.0: '+'}
    cooccurences = {}
    for x,y in aligned_fv:
        for i in range(len(x)):
            diffs = x[i] != y[i]
            errors = []
            if sum(diffs) <= 10: # More errors are likely to be noise (e.g. introduced by undetected metathesis)
                # Get the indices of the differing elements
                diff_indices = np.where(diffs)[0]
                for ind in diff_indices:
                    if x[i][ind] > 2:
                        continue
                    err_name = featnames[ind] + ": " + vals[y[i][ind]] + " to " + vals[x[i][ind]]
                    if err_name not in errors_fv:
                        errors_fv[err_name] = 0
                    errors_fv[err_name] += 1
                    # Additionally count the coocurrences of the errors
                    errors.append(featnames[ind])
            # Get all pairwise combinations of the errors
            for comb in combinations(errors, 2):
                if comb not in cooccurences:
                    cooccurences[comb] = 0
                cooccurences[comb] += 1
                
    
    # Sort the errors by frequency
    errors_fv = {k: v for k, v in sorted(errors_fv.items(), key=lambda item: item[1], reverse=True)}

    return errors_ipa, errors_fv, cooccurences

def cooccurrence_heatmap(filename, featnames):
    # Load the cooccurrences into a dataframe
    df = pd.read_csv(filename, sep="\t", index_col=0)
    
    # Create a pivot table from the dataframe, keeping the order from featnames
    df['Feature 1'] = pd.Categorical(df['Feature 1'], categories=featnames, ordered=True)
    df['Feature 2'] = pd.Categorical(df['Feature 2'], categories=featnames, ordered=True)
    df = df.sort_values(['Feature 1', 'Feature 2'])
    # Add symmetry
    df2 = df.copy()
    df2['Feature 1'] = df['Feature 2']
    df2['Feature 2'] = df['Feature 1']
    df2['proportion'] = df['proportion']
    df = pd.concat([df, df2])
    df = df.groupby(['Feature 1', 'Feature 2']).sum().reset_index()

    pivot_table = df.pivot(index='Feature 1', columns='Feature 2', values='proportion').fillna(0)
    
    # Create a heatmap from the pivot table
    plt.figure(figsize=(10, 10))
    hm = sns.heatmap(pivot_table, linewidth=.5, linecolor='#faf3f0', 
                     cmap=sns.light_palette("xkcd:copper", as_cmap=True))
    hm.xaxis.set_ticks_position('top')
    hm.xaxis.set_tick_params(rotation=90)
    plt.title('Error Cooccurrence Heatmap')
    plt.savefig(filename.replace('.csv', '_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()




############
proto_map = {'Burmish':'ProtoBurmish', 'Purus':'ProtoPurus',
                'Lalo':'ProtoLalo', 'Bai':'ProtoBai', 'Karen':'ProtoKaren',
                'Romance':'Latin'} # Map of proto-languages to language families
language_families = list(proto_map.keys())
model_name = "s2s_proto"
missing_props = ['0.8', '0.5', '0.1']
for missing_prop in missing_props: 
    ft_epchs = 20
    ft_lr = 5e-4
    if missing_prop == '0.8': 
        ft_bs = 16
    else:
        ft_bs = 48
    
    errors_ipa_family = {family: {} for family in language_families}
    errors_fv_family = {family: {} for family in language_families}
    coocc_errors_family = {family: {} for family in language_families}
    errors_ipa_all = {}
    errors_fv_all = {}
    coocc_errors_all = {}

    for family in tqdm(language_families):
        data = load_proto_data(proto_map, family)
        # Load fine-tuning data
        train_data = data[f'finetune_train_{missing_prop}_{1}']
        dev_data = data[f'finetune_dev_{missing_prop}_{1}']
        test_data = data[f'finetune_test_{missing_prop}_{1}']
        # Load vocab
        with open(f"./model_checkpoints/{model_name}_vocab.pkl", "rb") as f:
            char2featvec, char2idx = pickle.load(f)
        idx2char = {v: k for k, v in char2idx.items()}
        train_dataset = ProtoDataset(train_data, char2featvec, char2idx)
        train_dataloader = DataLoader(train_dataset, batch_size=ft_bs, collate_fn=collate_fn, shuffle = True)
        dev_dataset = ProtoDataset(dev_data, char2featvec, char2idx)
        dev_dataloader = DataLoader(dev_dataset, batch_size=ft_bs, collate_fn=collate_fn)
        test_dataset = ProtoDataset(test_data, char2featvec, char2idx)
        test_dataloader = DataLoader(test_dataset, batch_size=ft_bs, collate_fn=collate_fn)

        model = CognateS2S(
            char2idx,
            feat_dim=39,
            hidden_dim=128,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=2,
            dropout=0.15
        ).to(device)

        model.load_state_dict(torch.load(f"./model_checkpoints/{model_name}.pt"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr)
        with torch.no_grad():
            model = training_proto_finetune(model, 
                            train_dataloader, 
                            dev_dataloader, 
                            optimizer, 
                            ft_epchs,
                            model_name, 
                            save = False)
            
        # Error analysis
        errors_ipa, errors_fv, cooccurrences = error_analysis(model, test_dataloader, char2idx, idx2char)
        # Normalize the errors to proportions
        sum_ipa = sum(errors_ipa.values())
        sum_fv = sum(errors_fv.values())
        sum_coocc = sum(cooccurrences.values())
        for k,v in errors_ipa.items():
            errors_ipa[k] = round(v / sum_ipa, 4)
        for k,v in errors_fv.items():
            errors_fv[k] = round(v / sum_fv, 4)
        for k,v in cooccurrences.items():
            cooccurrences[k] = round(v / sum_coocc, 4)
        errors_ipa_family[family] = errors_ipa
        errors_fv_family[family] = errors_fv
        coocc_errors_family[family] = cooccurrences
        for k,v in errors_ipa.items():
            if k not in errors_ipa_all:
                errors_ipa_all[k] = 0
            errors_ipa_all[k] += v
        for k,v in errors_fv.items():
            if k not in errors_fv_all:
                errors_fv_all[k] = 0
            errors_fv_all[k] += v
        for k,v in cooccurrences.items():
            if k not in coocc_errors_all:
                coocc_errors_all[k] = 0
            coocc_errors_all[k] += v
    
    # Normalize again
    sum_ipa = sum(errors_ipa_all.values())
    sum_fv = sum(errors_fv_all.values())
    sum_coocc = sum(coocc_errors_all.values())
    for k,v in errors_ipa_all.items():
        errors_ipa_all[k] = round(v / sum_ipa, 4)
    for k,v in errors_fv_all.items():
        errors_fv_all[k] = round(v / sum_fv, 4)
    for k,v in coocc_errors_all.items():
        coocc_errors_all[k] = round(v / sum_coocc, 4)
    
    # Sort
    errors_ipa_all = {k: v for k, v in sorted(errors_ipa_all.items(), key=lambda item: item[1], reverse=True)}
    errors_fv_all = {k: v for k, v in sorted(errors_fv_all.items(), key=lambda item: item[1], reverse=True)}
    coocc_errors_all = {k: v for k, v in sorted(coocc_errors_all.items(), key=lambda item: item[1], reverse=True)}


    # Convert the error dicts to dfs, adding a column with the propotion of errors
    errors_ipa_all = pd.DataFrame.from_dict(errors_ipa_all, orient='index', columns=['proportion'])
    errors_ipa_all['gold'] = [x.split(' / ')[0] for x in errors_ipa_all.index]
    errors_ipa_all['prediction'] = [x.split(' / ')[1] for x in errors_ipa_all.index]
    errors_ipa_all = errors_ipa_all.reset_index(drop=True)
    errors_ipa_all = errors_ipa_all[['gold', 'prediction', 'proportion']]

    errors_fv_all = pd.DataFrame.from_dict(errors_fv_all, orient='index', columns=['proportion'])
    errors_fv_all['feature'] = [x.split(': ')[0] for x in errors_fv_all.index]
    errors_fv_all['gold'] = [x.split(': ')[1].split(' to ')[0] for x in errors_fv_all.index]
    errors_fv_all['prediction'] = [x.split(': ')[1].split(' to ')[1] for x in errors_fv_all.index]
    errors_fv_all = errors_fv_all.reset_index(drop=True)
    errors_fv_all = errors_fv_all[['feature', 'gold', 'prediction', 'proportion']]
    
    coocc_errors_all = pd.DataFrame.from_dict(coocc_errors_all, orient='index', columns=['proportion'])
    coocc_errors_all['Feature 1'] = [x[0] for x in coocc_errors_all.index]
    coocc_errors_all['Feature 2'] = [x[1] for x in coocc_errors_all.index]
    coocc_errors_all = coocc_errors_all.reset_index(drop=True)
    coocc_errors_all = coocc_errors_all[['Feature 1', 'Feature 2', 'proportion']]

    errors_ipa_all.to_csv(f"./error_analysis/{model_name}_IPA_errors_{missing_prop}.csv", sep="\t")
    errors_fv_all.to_csv(f"./error_analysis/{model_name}_FV_errors_{missing_prop}.csv", sep="\t")
    coocc_errors_all.to_csv(f"./error_analysis/{model_name}_coocc_errors_{missing_prop}.csv", sep="\t")
    
    for family in language_families:
        temp1 = pd.DataFrame.from_dict(errors_ipa_family[family], orient='index', columns=['proportion'])
        temp1['gold'] = [x.split(' / ')[0] for x in temp1.index]
        temp1['prediction'] = [x.split(' / ')[1] for x in temp1.index]
        temp1 = temp1.reset_index(drop=True)
        temp1 = temp1[['gold', 'prediction', 'proportion']]

        temp2 = pd.DataFrame.from_dict(errors_fv_family[family], orient='index', columns=['proportion'])
        temp2['feature'] = [x.split(': ')[0] for x in temp2.index]
        temp2['gold'] = [x.split(': ')[1].split(' to ')[0] for x in temp2.index]
        temp2['prediction'] = [x.split(': ')[1].split(' to ')[1] for x in temp2.index]
        temp2 = temp2.reset_index(drop=True)
        temp2 = temp2[['feature', 'gold', 'prediction', 'proportion']]

        coocc_temp = pd.DataFrame.from_dict(coocc_errors_family[family], orient='index', columns=['proportion'])
        coocc_temp['Feature 1'] = [x[0] for x in coocc_temp.index]
        coocc_temp['Feature 2'] = [x[1] for x in coocc_temp.index]
        coocc_temp = coocc_temp.reset_index(drop=True)
        coocc_temp = coocc_temp[['Feature 1', 'Feature 2', 'proportion']]
        # Save the errors to a file
        temp1.to_csv(f"./error_analysis/{model_name}_IPA_errors_{missing_prop}_{family}.csv", sep="\t")
        temp2.to_csv(f"./error_analysis/{model_name}_FV_errors_{missing_prop}_{family}.csv", sep="\t")
        coocc_temp.to_csv(f"./error_analysis/{model_name}_coocc_errors_{missing_prop}_{family}.csv", sep="\t")



# Create heatmaps for the cooccurrences
for missing_prop in missing_props:
    cooccurrence_heatmap(f"./error_analysis/{model_name}_coocc_errors_{missing_prop}.csv", featnames)




##
# Count errors that are gaps
missprops = ['0.8', '0.5', '0.1']
for missprop in missprops:
    # Load the errors from the file

    df = pd.read_csv(f"./error_analysis/s2s_proto_IPA_errors_{missprop}.csv", sep="\t", index_col=0)
    deletions = df[df['prediction'].str.contains('-')]
    insertions = df[df['gold'].str.contains('-')]
    # Sum the proportions of the gaps
    ndel = deletions['proportion'].sum().round(4)
    nins = insertions['proportion'].sum().round(4)
    ngaps = round(ndel + nins, 4)
    print(f"Test proportion {missprop}:")
    print(f"Number of deletions: {ndel}")
    print(f"Number of insertions: {nins}")
    print(f"Number of gaps: {ngaps}")
    print("")

# Count errors that are tones
missprops = ['0.8', '0.5', '0.1']
tones = ['¹', '²', '³', '⁴', '⁵']
for missprop in missprops:
    # Load the errors from the file
    df = pd.read_csv(f"./error_analysis/s2s_proto_IPA_errors_{missprop}.csv", sep="\t", index_col=0)
    tones_df = df[df['gold'].str.contains('|'.join(tones)) | df['prediction'].str.contains('|'.join(tones))]

    # Sum the proportions of the tones
    ntone = tones_df['proportion'].sum().round(4)
    print(f"Test proportion {missprop}:")
    print(f"Number of tones: {ntone}")
    print("")

# Count errors that are both vowels, both consonants or one of each or contain diphtongs
clts = CLTS("./data/clts/clts-2.3.0")
bipa = clts.bipa # broad IPA
sv_ipa = SoundVectors(ts=bipa)
missprops = ['0.8', '0.5', '0.1']
for missprop in missprops:
    # Load the errors from the file
    df = pd.read_csv(f"./error_analysis/s2s_proto_IPA_errors_{missprop}.csv", sep="\t", index_col=0)
    # Iterate over the errors and check if any of the two (x / y) are vowels based on the feature vector
    # A vowel is: sv_ipa['c'].cons == -1
    only_vowels = 0
    only_consonants = 0
    both = 0
    diphtongs = 0
    for i in range(len(df)):
        gold = df.iloc[i]['gold']
        pred = df.iloc[i]['prediction']
        if '-' in gold or '-' in pred:
            continue
        if (sv_ipa[gold].cons == -1) and (sv_ipa[pred].cons == -1):
            only_vowels += df.iloc[i]['proportion']
        elif (sv_ipa[gold].cons == 1) and (sv_ipa[pred].cons == 1):
            only_consonants += df.iloc[i]['proportion']
        elif (sv_ipa[gold].cons == -1) and (sv_ipa[pred].cons == 1):
            both += df.iloc[i]['proportion']
        elif (sv_ipa[gold].backshift != 0) or (sv_ipa[pred].backshift != 0):
            diphtongs += df.iloc[i]['proportion']
    # Calculate the proportion of vowels in the errors
    print(f"Test proportion {missprop}:")
    print(f"Number of only vowels: {only_vowels}")
    print(f"Number of only consonants: {only_consonants}")
    print(f"Number of both: {both}")
    print(f"Number of diphtongs: {diphtongs}")
    print("")

# Count number of consonants and vowels in the vocabulary
with open(f"./model_checkpoints/s2s_proto_vocab.pkl", "rb") as f:
        char2featvec, char2idx = pickle.load(f)
vocab = list(char2idx.keys())
vocab = [x for x in vocab if '[' not in x]

nvowels = 0
nconsonants = 0
for c in vocab:
    if sv_ipa[c].cons == -1:
        nvowels += 1
    elif sv_ipa[c].cons == 1:
        nconsonants += 1
print(f"Number of vowels in the vocabulary: {nvowels}")
print(f"Number of consonants in the vocabulary: {nconsonants}")
print(f"Total number of characters in the vocabulary: {len(vocab)}")
