from soundvectors import SoundVectors
from pyclts import CLTS
clts = CLTS("./data/clts/clts-2.3.0")
bipa = clts.bipa # broad IPA
sv_ipa = SoundVectors(ts=bipa)

def get_feature_vector(char, vec_len=39):
    if char == '[UNK]':
        return tuple([1] * vec_len) 
    if char == '[SEP]':
        return tuple([3] * vec_len)
    if char == '[PAD]':
        return tuple([4] * vec_len)

    featvec = sv_ipa([char])[0]
    if featvec:
        featvec = [x+1 for x in list(featvec)] # Must be positive for classification
        return tuple(featvec)
    else:
        return tuple([1] * vec_len) 
        
def tokenizer(vocab, langs):
    char2featvec = {char: get_feature_vector(char) for char in vocab}
    all = set(langs + vocab)
    char2idx = {char: i for i, char in enumerate(all, start=1)}
    return char2featvec, char2idx