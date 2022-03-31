import nltk
import numpy as np
import re
import pandas as pd
from nltk.parse.corenlp import CoreNLPDependencyParser

import opinion_lexicon as OP

def clean(text):
    """ Removes all special characters and numericals leaving the alphabets """
    text = re.sub(r"[()\"#/@;<>{}`+=~|!?]", "", text)
    text = re.sub(r"[.]", ". ", text)
    return text

def get_aspect(word, aspects_type):
    """ Find aspect topic of a word in dict aspects_type """
    if word in aspects_type:
        return word
    for key in aspects_type:
        if word in aspects_type[key]:
            return key


def get_polarity(dict_inf):
    """ Get the polarity of the sentence composed by the opinion_word and aspect_term in dict_inf """
    sentence = dict_inf["opinion_word"] + " " + dict_inf["aspect_term"]
    return OP.get_polarity(sentence)


def insert_just_before(dict_inf, item_type, word1, word2):
    """ Insert word2 just before than word1 in dict_inf[item_type] """
    word_list = dict_inf[item_type].split()
    idx = word_list.index(word1)
    new_split = [word2]
    new_split += word_list[idx:]
    dict_inf[item_type] = " ".join(word_list[:idx] + new_split)
    return dict_inf


def save_and_reset_information(data, dict_inf, aspect):
    """ Save new tuple in dataframe data if is an opinion word wich an aspect of the vocabulary dict. 
        Reset current info for deafult values. """
    if aspect != None and "opinion_word" in dict_inf:
        dict_inf["aspect"] = aspect.upper()
        dict_inf["polarity"] = get_polarity(dict_inf)
        data = data.append(dict_inf, ignore_index=True)

    dict_inf = {}
    aspect = None
    return data, dict_inf, aspect


def compute_rules(head, relation, dependent, data, dict_inf, aspect, aspects_type):
    """ Compute rules of grammar to find opinion_word and aspect_terms. """
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


def get_aspect_opinion(idx, review, aspects_type):
    """ Main code to get all the tuples from a review and save it in a dataframe """
    
    data = pd.DataFrame(columns=["aspect", "opinion_word", "aspect_term", "polarity"])
    dependency_parser = CoreNLPDependencyParser()

    review = clean(review)
    sentences = nltk.sent_tokenize(review)
    for s in sentences:
        (result,) = dependency_parser.raw_parse(s)

        dict_inf, aspect = {}, None
        for head, relation, dependent in result.triples():
            data, dict_inf, aspect = compute_rules(
                head, relation, dependent, data, dict_inf, aspect, aspects_type
            )
        data, _, _ = save_and_reset_information(data, dict_inf, aspect)
    data["review_id"] = idx
    data = data.set_index("review_id")
    return data
