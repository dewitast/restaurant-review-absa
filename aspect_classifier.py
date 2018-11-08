from nltk.corpus import wordnet as wn

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
        return ''

def nounify(word, pos):
    if pos == 'n':
        return word
    synsets = wn.synsets(word, pos=pos)
    lemmas = [l for s in synsets for l in s.lemmas() if s.pos() == pos]
    related = [(l, l.derivationally_related_forms()) for l in lemmas]
    related_noun = [l for r in related for l in r[1] if l.synset().pos() == 'n']
    words = [l.name() for l in related_noun]
    
    ln = len(words)
    result = ""
    prob = 0
    for w in set(words):
        if float(words.count(w))/ln > prob:
            result = w
            prob = float(words.count(w))/ln
            
    return result

def wu_palmer_similarity(s1, s2):
    hp1, hp2 = s1.hypernym_paths()[0], s2.hypernym_paths()[0]
    d1, d2 = len(hp1), len(hp2)

    d = min(d1, d2)
    while (hp1[d-1]!=hp2[d-1] and d > 0):
            d -= 1
    
    return float(2 * d) / (d1 + d2)

aspects = ["food", "price", "ambience", "service", "place"]
test = nounify("steak", "n")

for aspect in aspects:
    sa = wn.synsets(aspect)
    st = wn.synsets(test)
    maks = 0
    for x in sa:
        for y in st:
            if x.pos() == y.pos() and x.pos() == 'n':
                maks = max(maks, wu_palmer_similarity(x, y, False))
    print(aspect + ' ' + str(maks))