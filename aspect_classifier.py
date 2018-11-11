from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import string

def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('VB'):
        return wn.VERB
    elif tag.startswith('JJ'):
        return wn.ADJ
    elif tag.startswith('RB'):
        return wn.ADV
    else:
        return wn.NOUN

def nounify(word, pos):
    if pos == wn.NOUN:
        return word
    synsets = wn.synsets(word, pos=pos)
    lemmas = [l for s in synsets for l in s.lemmas() if s.pos() == pos]
    related = [(l, l.derivationally_related_forms()) for l in lemmas]
    related_noun = [l for r in related for l in r[1] if l.synset().pos() == wn.NOUN]
    words = [l.name() for l in related_noun]
    
    ln = len(words)
    result = ""
    prob = 0
    for w in set(words):
        if float(words.count(w))/ln > prob:
            result = w
            prob = float(words.count(w))/ln
            
    if (len(result) == 0):
        return word
    return result

def wu_palmer_similarity_synset(s1, s2):
    hp1, hp2 = s1.hypernym_paths()[0], s2.hypernym_paths()[0]
    d1, d2 = len(hp1), len(hp2)

    d = min(d1, d2)
    while (hp1[d-1]!=hp2[d-1] and d > 0):
            d -= 1
    
    return float(2 * d) / (d1 + d2)

def wu_palmer_similarity(str1, pos1, str2, pos2):
    str1 = nounify(str1, pos1)
    str2 = nounify(str2, pos2)
    
    s1 = wn.synsets(str1)
    s2 = wn.synsets(str2)
    maks = 0
    for x in s1:
        for y in s2:
            if x.pos() == y.pos() and x.pos() == wn.NOUN:
                maks = max(maks, wu_palmer_similarity_synset(x, y))
                
    return maks

def get_aspects(doc):
    aspects = ["food", "price", "service", "place"]
    key_words = {
        "food": ["food", "drink", "menu", "deliciousness", "tastiness", "spiciness"],
        "price": ["price", "expensiveness", "cheapness", "money"],
        "place": ["place", "table", "chair", "seating", "parking", "vibe", "music", "decoration", "ambience"],
        "service": ["service", "waiter", "waitress", "owner", "manager", "serving"]
    }
    total = {
        "food": 0,
        "price": 0,
        "place": 0,
        "service": 0,
    }
    cnt = 0
    for word in doc:
        if word[2] == 'B':
            cnt += 1
        if word[2] != 'O' or wordnet_pos_code(word[1]) == wn.ADJ:
            best_aspect, maks = '', 0
            for aspect in aspects:
                maks_tmp = 0
                for key in key_words[aspect]:
                    tmp = wu_palmer_similarity(word[0], wordnet_pos_code(word[1]), key, wn.NOUN)
                    if tmp > maks_tmp:
                        maks_tmp = tmp
                print(word[0] + ' - ' + aspect + ' : ' + str(maks_tmp))
                if maks_tmp > maks:
                    maks = maks_tmp
                    best_aspect = aspect
                    
            if maks >= 0.5:
                total[best_aspect] = total[best_aspect] + 1
                
    if cnt == 0:
        cnt = 1
    if cnt > 4:
        cnt = 4
    
    lst = [(c, a) for a, c in total.items()]
    lst.sort()
    return lst[4-cnt:]