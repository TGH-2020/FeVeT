"""
The data loading functions are heavily based on the data retrieval functions from https://doi.org/10.18653/v1/2023.emnlp-main.423.
See: https://github.com/mahesh-ak/CognateTransformer/blob/main/src/data_load.py
The data (and folds for proto) is also the same and found here: https://github.com/mahesh-ak/CognateTransformer/tree/main/data
"""


import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import *
from itertools import chain
import json
from sklearn.model_selection import train_test_split


## Data retrieval

# Transforms a row of the dataframe into a dictionary
def get_dict(row, sol_col = None):
    cogset = {}
    proto_sol = None
    if sol_col is not None:
        sol_col = "[" + sol_col + "]"

    for key, val in row.items():
        if key == 'COGID' or val in ['-']:
            continue
        key = '[' + key + ']'
        if val in ['?']:
            if '_sol' in key:
                cogset[key.replace('_sol','')] = val
            else:
                cogset[key] = val
        else:
            val = val.replace('+', '')
            if '_sol' in key:
                cogset[key.replace('_sol','')] = val.split()
            elif key == sol_col:
                proto_sol = {key: val.split()}
            else:
                cogset[key] = val.split()
    
    if sol_col is not None:
        return cogset, proto_sol
    else:
        return cogset

# Load the reflex data
def load_reflex_data(specific_file = None):
    files = {'train-0.10':[], 'train-0.30': [], 'train-0.50': [], 
            'test-0.10':[], 'test-0.30':[], 'test-0.50': []}
    if specific_file is not None:
        subdirs = os.listdir(specific_file)
        for f in subdirs:
                if 'training' in f and 'surprise' in specific_file:
                    prop = f.replace('training-','').replace('.tsv','')
                    files[f"train-{prop}"].append(os.path.join(specific_file,f))
                if 'cognates' in f and 'surprise' not in specific_file:
                    for prop in ['0.10','0.30','0.50']:
                        files[f"train-{prop}"].append(os.path.join(specific_file,f))
                if 'test' in f and 'surprise' in specific_file:
                    prop = f.replace('test-','').replace('.tsv','')
                    sol_f = f"solutions-{prop}.tsv"
                    files[f"test-{prop}"].append((os.path.join(specific_file,f), os.path.join(specific_file,sol_f)))
    else:
        data_paths = ["./data/reflex_data/data-surprise/", "./data/reflex_data/data/"]

        for data_path in data_paths:
            dirs = os.listdir(data_path)
            for fd in dirs:
                if '.' in fd:
                    continue
                sub_path = os.path.join(data_path,fd)
                subdirs = os.listdir(sub_path)
                for f in subdirs:
                    if 'training' in f and 'surprise' in data_path:
                        prop = f.replace('training-','').replace('.tsv','')
                        files[f"train-{prop}"].append(os.path.join(sub_path,f))
                    if 'cognates' in f and 'surprise' not in data_path:
                        for prop in ['0.10','0.30','0.50']:
                            files[f"train-{prop}"].append(os.path.join(sub_path,f))
                    if 'test' in f and 'surprise' in data_path:
                        prop = f.replace('test-','').replace('.tsv','')
                        sol_f = f"solutions-{prop}.tsv"
                        files[f"test-{prop}"].append((os.path.join(sub_path,f), os.path.join(sub_path,sol_f)))
    
    data =  {'train-0.10': {'data':[]},
            'train-0.30': {'data':[]},
            'train-0.50': {'data':[]},
            'test-0.10': {'data':[], 'solns':[]}, 
            'test-0.30': {'data':[], 'solns':[]},
            'test-0.50': {'data':[], 'solns':[]}}

    for key, f_list in tqdm(files.items()):
        if 'train' in key:
            for f in f_list:
                df = pd.read_csv(f, sep='\t')
                df.fillna('-', inplace=True)
                data[key]['data'] += df.apply(lambda x: get_dict(x), axis=1).tolist()
        else: ## test
            for f in f_list:
                df_test = pd.read_csv(f[0], sep='\t')
                df_test.fillna('-', inplace= True)
                df_sol = pd.read_csv(f[1], sep='\t')
                df_sol.fillna('-', inplace= True)
                df_sol.rename(columns={col: col+'_sol' for col in list(df_sol.columns) if col != 'COGID'}, inplace=True)
                df_test = df_test.merge(df_sol, on= 'COGID')
                df_sol = df_test.drop(columns=[col for col in list(df_test.columns) if '_sol' not in col and col !='COGID'])
                df_test = df_test.drop(columns=[col for col in list(df_test.columns) if '_sol' in col and col !='COGID'])
                data[key]['data'] += df_test.apply(lambda x: get_dict(x), axis=1).tolist()
                data[key]['solns'] += df_sol.apply(lambda x: get_dict(x), axis=1).tolist()
    
    return data



def Merge(dict_lst):
        out = {}
        for D in dict_lst:
            out.update(D)
        return out

def load_proto_data(proto_map, specific_family = None, valid_lim = 10, load_pretrain_data = True):
    data_paths = ["./data/reflex_data/data-surprise/", "./data/reflex_data/data/"]
    files = []
    print("Loading proto reconstruction data...")
    
    # Get all filenames
    for data_path in data_paths:
        dirs = os.listdir(data_path)
        for fd in dirs:
            if '.' in fd:
                continue
            sub_path = os.path.join(data_path,fd)
            subdirs = os.listdir(sub_path)
            for f in subdirs:
                if 'cognates' in f:
                    files.append(os.path.join(sub_path,f))

    # 10fold cross validation on finetuning data (pre-prepared)
    data =  {'pretrain': {'data':[]}}
    for prop in ['0.1', '0.5', '0.8']:
        for valid in range(valid_lim):
            data[f"finetune_train_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}
            data[f"finetune_test_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}
            data[f"finetune_dev_{prop}_{valid + 1}"] = {'data':[], 'solns':[]}
    
    # Load pre-training data, remove proto-languages
    if specific_family is None and load_pretrain_data:
        for f in tqdm(files):
            df = pd.read_csv(f, sep='\t')
            df.fillna('-', inplace=True)
            for lng, plng in proto_map.items(): # Remove from pretraining if it is a proto-lang
                if plng in df.columns:
                    df.drop(columns=[plng], inplace=True)
            # We filter out single entries that remain after removing proto-languages
            data['pretrain']['data'] += df.apply(lambda x: get_dict(x) if (x != "-").sum() > 2 else None, axis=1).tolist()
            data['pretrain']['data'] = [x for x in data['pretrain']['data'] if x is not None]
    
    # Create finetuning data consisting of train, dev, and test sets
    # for each proportion and for each fold
    for prop in tqdm(['0.1','0.5', '0.8']):
    
        train_path = f"./data/proto_data/data-{prop}/testlists/"
        test_path = f"./data/proto_data/data-{prop}/testitems/" 

        for file_num in range(1,valid_lim+1):
            dirs = os.listdir(train_path)
            for fd in dirs:
                if '.' in fd:
                    continue
                if specific_family is not None and fd != specific_family:
                    continue
                df_ = {}
                f_str = "test-{0}".format(file_num)
    
                sub_path = os.path.join(test_path,fd)    
                f_pth = os.path.join(sub_path, f_str+'.json')
                f_json = json.load(open(f_pth,'r'))
                rows = []
                for line in f_json:
                    row = {proto_map[fd]: ' '.join(line[1])}
                    for word, lng in zip(line[2],line[3]):
                        row[lng] = ' '.join(word)
                    rows.append(row)
                df_['test'] = pd.DataFrame(rows).fillna('-')
    
                sub_path = os.path.join(train_path,fd)  
  
                f_pth = os.path.join(sub_path, f_str+'.tsv')
                df = pd.read_csv(f_pth, sep= '\t')
                df.drop(columns=['FORM', 'ID', 'ALIGNMENT', 'CONCEPT'], inplace=True)
                df['dict'] = df.apply(lambda x: {x['DOCULECT']: x['TOKENS']}, axis=1)
                df.drop(columns=['DOCULECT', 'TOKENS'], inplace=True)
                df = df.groupby('COGID').agg(Merge).reset_index()
                rows = df['dict'].tolist()
                df = pd.DataFrame(rows).fillna('-')
                df_['train'], df_['dev'] = train_test_split(df, test_size= 0.08)
    
                for div in ['train', 'dev', 'test']:
                    dat = df_[div].apply(lambda x: get_dict(x, sol_col= proto_map[fd]), axis=1).tolist()
                    sol = [row[1] for row in dat]
                    dat = [row[0] for row in dat]
                    data[f"finetune_{div}_{prop}_{file_num}"]['data'] += dat
                    data[f"finetune_{div}_{prop}_{file_num}"]['solns'] += sol
                df = pd.concat([df, df_['test']])
                df.drop(columns=[proto_map[fd]], inplace=True)
                
                # We add one fold of our finetuning data with high coverage to the pretraining data (proto-lang removed)
                if prop == '0.1' and file_num == 1:
                    data['pretrain']['data'] += df.apply(lambda x: get_dict(x), axis=1).tolist()

    return data



def get_vocab(train_data, key = 'data'):
    vocab = set()
    langs = set()
    for row in train_data[key]:
        chars = set(chain(*row.values()))
        vocab = vocab.union(chars)
        langs = langs.union(row.keys())
    vocab = vocab.union(['[UNK]', '[SEP]', '[PAD]'])
    # Prepend [PAD] to langs
    langs = ['[PAD]'] + list(langs)
    return list(vocab), langs



########################################################
## Dataset preparation and collate function

# Input of form:
# Cognates: 
# A B C [SEP] [PAD]
# A B [SEP] [PAD] [PAD]
# A B C [SEP] [PAD] 

# Langs: 
# 1 1 1 1 0
# 2 2 2 0 0
# 3 3 3 3 0 

# Build a torch dataset
class TrainDataset(IterableDataset):
    def __init__(self, data, char2featvec, all2id):
        super().__init__()
        self.data = data['data']
        self.char2featvec = char2featvec
        self.all2id = all2id # for labels only
        self.pad_token = char2featvec['[PAD]']
        self.unk_token = char2featvec['[UNK]']
    
    def __iter__(self):
        # Randomly mask 1 value from the data
        while True:
            for cogset in self.data:
                langs = list(cogset.keys())
                target_langs = torch.randint(0, len(langs), (1,)).item()
                target_langs = langs[target_langs]
                target = cogset[target_langs]
                # Remove masked_key from langs
                langs.remove(target_langs)

                # Change the order of the languages randomly
                rand_idx = torch.randperm(len(langs)).tolist()
                langs = [langs[i] for i in rand_idx]

                valids = [cogset[key] for key in langs]
          
                # Repeat each lang as many times as the length of its corresponding sequence
                langs = list(chain(*[[[key]*(len(val)+1)] for key, val in zip(langs, valids)]))

                # Flatten and add '[SEP]' to the end of each valid
                valids = list(chain(*[[[char for char in valid] + ['[SEP]']] for valid in valids]))

                max_len = max([len(valid) for valid in valids])

                # Encode the data; ressort to [UNK] if char not in vocab
                langs = [[self.all2id[lang] for lang in lang] for lang in langs]
                target_langs = self.all2id[target_langs]
                valids = [[self.char2featvec.get(char, self.unk_token) for char in valid] for valid in valids]
                target = target + ['[SEP]']
                targets = [self.all2id.get(char, self.all2id['[UNK]']) for char in target]
                targets_featvec = [self.char2featvec.get(char, self.unk_token) for char in target]

                # Pad langs with 0 and valids with self.pad_token
                langs = [lang + [0]*(max_len-len(lang)) for lang in langs]
                valids = [valid + [self.pad_token]*(max_len-len(valid)) for valid in valids]

                langs = torch.tensor(langs)
                target_langs = torch.tensor(target_langs)
                valids = torch.tensor(valids, dtype=torch.float32)
                targets = torch.tensor(targets)
                targets_featvec = torch.tensor(targets_featvec, dtype=torch.float32)

                yield langs, valids, target_langs, targets, targets_featvec

# Similar to TrainDataset, but for evaluation. That is, we do not need an iterator.
class DevDataset(Dataset):
    def __init__(self, data, char2featvec, all2id):
        super().__init__()
        self.data = data['data']
        self.char2featvec = char2featvec
        self.all2id = all2id # for labels only
        self.pad_token = char2featvec['[PAD]']
        self.unk_token = char2featvec['[UNK]']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cogset = self.data[idx]
        langs = list(cogset.keys())
        target_langs = torch.randint(0, len(langs), (1,), generator=torch.Generator().manual_seed(idx)).item()
        target_langs = langs[target_langs]
        target = cogset[target_langs]
        langs.remove(target_langs)

        valids = [cogset[key] for key in langs]
        
        langs = list(chain(*[[[key]*(len(val)+1)] for key, val in zip(langs, valids)]))
        valids = list(chain(*[[[char for char in valid] + ['[SEP]']] for valid in valids]))

        max_len = max([len(valid) for valid in valids])
        
        # Encode the data; ressort to [UNK] if char not in vocab
        langs = [[self.all2id[lang] for lang in lang] for lang in langs]
        target_langs = self.all2id[target_langs]
        valids = [[self.char2featvec.get(char, self.unk_token) for char in valid] for valid in valids]
        target = target + ['[SEP]']
        targets = [self.all2id.get(char, self.all2id['[UNK]']) for char in target]
        targets_featvec = [self.char2featvec.get(char, self.unk_token) for char in target]

        # Pad langs with 0 and valids with self.pad_token
        langs = [lang + [0]*(max_len-len(lang)) for lang in langs]
        valids = [valid + [self.pad_token]*(max_len-len(valid)) for valid in valids]

        # Convert to tensors
        langs = torch.tensor(langs)
        target_langs = torch.tensor(target_langs)
        valids = torch.tensor(valids, dtype=torch.float32)
        targets = torch.tensor(targets)
        targets_featvec = torch.tensor(targets_featvec, dtype=torch.float32)
        
        return langs, valids, target_langs, targets, targets_featvec


class TestDataset(Dataset):
    def __init__(self, data, char2featvec, all2id):
        super().__init__()
        self.data = data['data']
        self.label = data['solns']
        self.char2featvec = char2featvec
        self.all2id = all2id # for labels only
        self.pad_token = char2featvec['[PAD]']
        self.unk_token = char2featvec['[UNK]']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cogset = self.data[idx]
        label = self.label[idx]
        langs = list(cogset.keys())
        target_langs = list(label.keys())[0]
        target = label[target_langs]
        langs.remove(target_langs)

        valids = [cogset[key] for key in langs]

        langs = list(chain(*[[[key]*(len(val)+1)] for key, val in zip(langs, valids)]))
        valids = list(chain(*[[[char for char in valid] + ['[SEP]']] for valid in valids]))
        
        max_len = max([len(valid) for valid in valids])
        
        langs = [[self.all2id[lang] for lang in lang] for lang in langs]
        target_langs = self.all2id[target_langs]
        valids = [[self.char2featvec.get(char, self.unk_token) for char in valid] for valid in valids]
        target = target + ['[SEP]']
        targets = [self.all2id.get(char, self.all2id['[UNK]']) for char in target]
        target_featvec = [self.char2featvec.get(char, self.unk_token) for char in target]

        # Pad langs with 0 and valids with self.pad_token
        langs = [lang + [0]*(max_len-len(lang)) for lang in langs]
        valids = [valid + [self.pad_token]*(max_len-len(valid)) for valid in valids]

        langs = torch.tensor(langs)
        target_langs = torch.tensor(target_langs)
        valids = torch.tensor(valids, dtype=torch.float32)
        targets = torch.tensor(targets)
        target_featvec = torch.tensor(target_featvec, dtype=torch.float32)

        return langs, valids, target_langs, targets, target_featvec



# Similar to the above, but for the fine-tuning data in the proto reconstruction task
class ProtoDataset(Dataset):
    def __init__(self, data, char2featvec, all2id):
        super().__init__()
        self.data = data['data']
        self.label = data['solns']
        self.char2featvec = char2featvec
        self.all2id = all2id # for labels only
        self.pad_token = char2featvec['[PAD]']
        self.unk_token = char2featvec['[UNK]']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            cogset = self.data[idx]
            label = self.label[idx]
            langs = list(cogset.keys())
            target_langs = list(label.keys())[0]
            target = label[target_langs]

            valids = [cogset[key] for key in langs]

            langs = list(chain(*[[[key]*(len(val)+1)] for key, val in zip(langs, valids)]))
            valids = list(chain(*[[[char for char in valid] + ['[SEP]']] for valid in valids]))
            
            # As we are using a 2D model, we need to pad the sequences to the same length
            max_len = max([len(valid) for valid in valids])
            
            langs = [[self.all2id[lang] for lang in lang] for lang in langs]
            target_langs = self.all2id[target_langs]
            valids = [[self.char2featvec.get(char, self.unk_token) for char in valid] for valid in valids]
            target = target + ['[SEP]']
            targets = [self.all2id.get(char, self.all2id['[UNK]']) for char in target]
            target_featvec = [self.char2featvec.get(char, self.unk_token) for char in target]

            # Pad langs with 0 and valids with self.pad_token
            langs = [lang + [0]*(max_len-len(lang)) for lang in langs]
            valids = [valid + [self.pad_token]*(max_len-len(valid)) for valid in valids]

            langs = torch.tensor(langs)
            target_langs = torch.tensor(target_langs)
            valids = torch.tensor(valids, dtype=torch.float32)
            targets = torch.tensor(targets)
            target_featvec = torch.tensor(target_featvec, dtype=torch.float32)

            return langs, valids, target_langs, targets, target_featvec


def collate_fn(batch):
    """
    Collate function to dynamically pad sequences in a batch.
    Args:
        batch: List of tuples (langs, valids, target_langs, target).
    Returns:
        Padded tensors for langs, valids, target_langs, and target.
    """
    # Unpack the batch
    langs, valids, target_langs, target, target_featvec = zip(*batch)

    # Get max number of rows (cognates) and cols (characters) in the batch
    max_rows = max([lang.size(0) for lang in langs])
    max_cols = max([lang.size(1) for lang in langs])
    feat_dim = valids[0].size(-1)
    batch_size = len(langs)

    # Pad langs with 0 and valids with pad_token, along both dimensions
    pad_token = get_feature_vector('[PAD]')[0]
    valids_padded = torch.full((batch_size, max_rows, max_cols, feat_dim), pad_token, dtype=torch.float32) 
    langs_padded = torch.zeros((batch_size, max_rows, max_cols), dtype=torch.long)
    for i, (lang, valid) in enumerate(zip(langs, valids)):
        rows, cols = lang.size()
        langs_padded[i, :rows, :cols] = lang
        valids_padded[i, :rows, :cols, :] = valid

    target_featvec_padded = pad_sequence(target_featvec, batch_first=True, padding_value=pad_token)

    # Pad target (IPA IDs)
    target_padded = pad_sequence(target, batch_first=True, padding_value=0) 

    # Stack target_langs
    target_langs_stacked = torch.stack(target_langs).unsqueeze(-1)  # Add a dimension for the sequence length

    return langs_padded, valids_padded, target_langs_stacked, target_padded, target_featvec_padded
   



