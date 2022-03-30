import nltk
import numpy as np
import re
import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser

nltk.download("opinion_lexicon")
nltk.download("sentiwordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

negativeWords = opinion_lexicon.negative()
positiveWords = opinion_lexicon.positive()

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import VaderConstants


def get_vader_score(sent):
    sid = SentimentIntensityAnalyzer()
    # Polarity score returns dictionary
    ss = sid.polarity_scores(sent)
    # Index ordered by polarity value
    idx = np.argmax(list(ss.values())[:-1])
    if idx == 0:
        return -1.0
    if idx == 1:
        return 0.0
    if idx == 2:
        return 1.0


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith("J"):
        return wn.ADJ
    elif tag.startswith("N"):
        return wn.NOUN
    elif tag.startswith("R"):
        return wn.ADV
    elif tag.startswith("V"):
        return wn.VERB
    return None


def get_sentiment(sentence):
    lemmatizer = WordNetLemmatizer()
    token = nltk.word_tokenize(sentence)
    after_tagging = nltk.pos_tag(token)
    sentiment = 0
    for word, tag in after_tagging:

        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
    return sentiment


# Define a function to clean the text
def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub(r"[()\"#/@;<>{}`+=~|!?]", "", text)
    text = re.sub(r"[.]", ". ", text)
    return text


def get_aspect(word, aspects_type):
    if word in aspects_type:
        return word
    for key in aspects_type:
        if word in aspects_type[key]:
            return key


def get_polarity(dict_inf):
    sentence = dict_inf["opinion_word"] + " " + dict_inf["aspect_term"]
    polarity = get_vader_score(sentence)
    if polarity == 0.0:
        if get_sentiment(sentence) < 0.0:
            return -1.0
        elif get_sentiment(sentence) > 0.0:
            return 1.0
        else:
            return 0.0
    return polarity


# AUXILIAR FUNCTIONS
def insert_just_before(dict_inf, item_type, word1, word2):
    word_list = dict_inf[item_type].split()
    idx = word_list.index(word1)
    new_split = [word2]
    new_split += word_list[idx:]
    dict_inf[item_type] = " ".join(word_list[:idx] + new_split)
    return dict_inf


def save_and_reset_information(data, dict_inf, aspect):
    if aspect != None and "opinion_word" in dict_inf:
        dict_inf["aspect"] = aspect.upper()
        dict_inf["polarity"] = get_polarity(dict_inf)
        # save new info
        data = data.append(dict_inf, ignore_index=True)

    # reset
    dict_inf = {}
    aspect = None
    return data, dict_inf, aspect


# MAIN CODE RULES
def get_information(head, relation, dependent, data, dict_inf, aspect, aspects_type):
    word1, pos1 = head
    word2, pos2 = dependent

    word1 = word1.lower()
    word2 = word2.lower()

    if relation == "amod" and pos1.startswith("NN") and pos2.startswith("JJ"):
        if "opinion_word" in dict_inf and word1 not in dict_inf["aspect_term"].split():
            data, dict_inf, aspect = save_and_reset_information(data, dict_inf, aspect)

        if pos1 == "NN" and aspect == None:
            aspect = get_aspect(word1, aspects_type)

        if "opinion_word" in dict_inf:
            dict_inf["opinion_word"] += " " + word2
        else:
            dict_inf["opinion_word"] = word2

        if "aspect_term" not in dict_inf:
            dict_inf["aspect_term"] = word1

    elif relation == "compound" and pos1.startswith("NN") and pos2.startswith("NN"):
        if "aspect_term" in dict_inf and word1 in dict_inf["aspect_term"].split():
            dict_inf = insert_just_before(dict_inf, "aspect_term", word1, word2)

            if aspect == None:
                aspect = get_aspect(word2, aspects_type)

    elif (
        relation == "advmod"
        and (pos1.startswith("JJ") or pos1.startswith("CD"))
        and pos2.startswith("RB")
    ):
        if "opinion_word" in dict_inf and word1 in dict_inf["opinion_word"].split():
            dict_inf = insert_just_before(dict_inf, "opinion_word", word1, word2)

    elif relation == "fixed" and pos1.startswith("RB") and pos2.startswith("RB"):
        if "opinion_word" in dict_inf and word1 in dict_inf["opinion_word"].split():
            dict_inf = insert_just_before(dict_inf, "opinion_word", word1, word2)

    elif relation == "conj" and pos1.startswith("JJ") and pos2.startswith("JJ"):
        if "opinion_word" in dict_inf and word1 in dict_inf["opinion_word"].split():
            dict_inf["opinion_word"] += ", " + word2

    elif relation == "nsubj" and pos2.startswith("NN"):
        if "opinion_word" in dict_inf and word1 not in dict_inf["aspect_term"].split():
            data, dict_inf, aspect = save_and_reset_information(data, dict_inf, aspect)

        if pos1.startswith("NN"):
            dict_inf["aspect_term"] = word1 + " " + word2
            aspect = get_aspect(word2, aspects_type)
        elif pos1.startswith("JJ") or pos1.startswith("CD"):
            dict_inf["aspect_term"] = word2
            dict_inf["opinion_word"] = word1
            aspect = get_aspect(word2, aspects_type)

    return data, dict_inf, aspect


def analyze_review(idx, review, aspects_type):
    data = pd.DataFrame(columns=["aspect", "opinion_word", "aspect_term", "polarity"])
    dependency_parser = CoreNLPDependencyParser()
    text = review
    review = clean(review)
    sentences = nltk.sent_tokenize(review)
    for s in sentences:
        (result,) = dependency_parser.raw_parse(s)

        dict_inf, aspect = {}, None
        for head, relation, dependent in result.triples():
            data, dict_inf, aspect = get_information(
                head, relation, dependent, data, dict_inf, aspect, aspects_type
            )
        data, _, _ = save_and_reset_information(data, dict_inf, aspect)
    data["review_id"] = int(idx)
    data["text"] = text
    return data
