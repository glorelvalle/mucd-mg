import nltk
import csv

nltk.download("opinion_lexicon")
nltk.download("sentiwordnet")


from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import VaderConstants
from nltk.sentiment import SentimentIntensityAnalyzer

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
    sid = SentimentIntensityAnalyzer()

    polarity = 1.0
    for word, tag in after_tagging:
        word = word.lower()
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV): continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma: continue
            
        # modifiers ('boosters') given in the NTLK Vader sentiment opinion module and modifier.csv
        if word in constants:
            boost = float(constants[word])
            if boost > 0.0 and polarity > 0.0: polarity += boost
            elif boost > 0.0 and polarity < 0.0: polarity -= boost
            elif boost < 0.0 and polarity > 0.0: 
                polarity += abs(boost)
                polarity *= -1
            elif boost < 0.0 and polarity < 0.0:
                polarity -= boost
                polarity *= -1
            continue            

        # Polarity score returns dictionary
        ss_compound = sid.polarity_scores(lemma)['compound']
        if ss_compound == 0.0:
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            # Take the first sense, the most common
            swn_synset = swn.senti_synset(synsets[0].name())
            # Compute different between positive and negative
            ss_compound = swn_synset.pos_score() - swn_synset.neg_score()
        
        if ss_compound != 0.0: polarity *= ss_compound
        if polarity>0.0: polarity += abs(ss_compound)
        if polarity<0.0: polarity -= abs(ss_compound)
         
              
    return polarity

def get_polarity(sentence):
    polarity = get_polarity_score(sentence)
    if polarity > 0.0: return 1.0
    elif polarity < 0.0: return -1.0
    else: return 0.0