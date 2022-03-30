import nltk
import csv

nltk.download("opinion_lexicon")
nltk.download("sentiwordnet")


from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import VaderConstants
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn


def load_modifiers():
    reader = csv.reader(open('modifiers/modifiers.csv', 'r'))
    modifier = {}
    for (k,v) in reader: modifier[k] = v
    return modifier

def extend_vader(constants, modifiers):
    for key_modifier in modifiers:
        if key_modifier not in constants:
            constants[key_modifier] = modifiers[key_modifier]
    return constants

def penn_to_wn(tag):
    if tag.startswith("J"): return wn.ADJ
    elif tag.startswith("N"): return wn.NOUN
    elif tag.startswith("R"): return wn.ADV
    elif tag.startswith("V"): return wn.VERB
    return None

def get_polarity_score(sentence):
    lemmatizer = WordNetLemmatizer()
    token = nltk.word_tokenize(sentence)
    after_tagging = nltk.pos_tag(token)
    constants = VaderConstants().BOOSTER_DICT
    modifiers = load_modifiers()
    constants = extend_vader(constants, modifiers)
    
    polarity = 1
    modifier = 0
    
    for word, tag in after_tagging:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV): continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma: continue
            
        synsets = wn.synsets(lemma, pos=wn_tag)
        
        # modifiers ('boosters') given in the NTLK Vader sentiment opinion module and modifier.csv
        if not synsets and word in constants:
            modifier += float(constants[word])
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        # Compute different between positive and negative
        diff = swn_synset.pos_score() - swn_synset.neg_score()
        if diff != 0: polarity *= diff
        if polarity>0.0: polarity += abs(diff)
        if polarity<0.0: polarity -= abs(diff)
            
    # Modify sentence sentiment regarding polarity
    if polarity < 0.0: polarity -= abs(modifier)
    elif polarity > 0.0:
        if modifier > 0.0: polarity += modifier
        elif modifier < 0.0: polarity = polarity*modifier - abs(modifier)
    else: polarity += modifier
        
    return polarity

def get_polarity(sentence):
    polarity = get_polarity_score(sentence)
    if polarity > 0.0: return 1.0
    elif polarity < 0.0: return -1.0
    else: return 0.0