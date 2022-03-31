import nltk
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("omw-1.4")
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()


def extend_aspects(aspects):
    """Extend aspect vocabulary with synsets wordnet"""
    for key in aspects:
        synsets = wn.synsets(key)
        for synset in synsets:
            lemmas = synset.lemma_names()
            aspects[key] = list(set(aspects[key] + lemmas))
    return aspects


def clean_nlp(review_text):
    """Cleans all punctuations and mayus from a given text"""
    review_text = review_text.replace("<br />", " ")
    review_text = review_text.replace("\[?\[.+?\]?\]", " ")
    review_text = review_text.replace("\/{3,}", " ")
    review_text = review_text.replace("\&\#.+\&\#\d+?;", " ")
    review_text = review_text.replace("\d+\&\#\d+?;", " ")
    review_text = review_text.replace("\&\#\d+?;", " ")
    review_text = review_text.replace("\:\|", "")
    review_text = review_text.replace("\:\)", "")
    review_text = review_text.replace("\:\(", "")
    review_text = review_text.replace("\:\/", "")
    review_text = review_text.replace("\s{2,}", " ")
    review_text = review_text.lower()
    return review_text


def rule1(doc):
    """Sentiment modifier + Aspect, assuming m is a child of a with an amod relationship"""
    rule = []
    for token in doc:
        m, a = "", ""
        if token.dep_ == "amod" and not token.is_stop:
            m = token.text
            a = token.head.text
            m_sub = token.children
            for sub in m_sub:
                if sub.dep_ == "advmov":
                    m = sub.text + " " + m
                    break
            a_sub = token.head.children
            for sub in a_sub:
                if sub.dep_ == "det" and sub.text == "no":
                    m = "not" + " " + m
                    break
        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(token.text)["compound"], 1))
    return rule


def rule2(doc):
    """Sentiment modifier + Aspect, assuming a is a child of somethinf that is a nsubj, and m is a child of something with dobj"""
    rule = []
    for token in doc:
        children = token.children
        m, a = "", ""
        prx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                a = child.text

            if (child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop:
                m = child.text

            if child.dep_ == "neg":
                n_prx = child.text
                prx = True

    if prx and m != "":
        m = n_prx + " " + m
        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(m)["compound"], 2))
    return rule


def rule3(doc):
    """Sentiment modifier + Aspect, assuming a is a child with relationship of nsubj and m is acomp"""
    rule = []
    for token in doc:
        children = token.children
        a, m = "", ""
        prx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                a = child.text

            if child.dep_ == "acomp" and not child.is_stop:
                m = child.text

            if child.dep_ == "aux" and child.tag_ == "MD":
                n_prx = "not"
                prx = True

            if child.dep_ == "neg":
                n_prx = child.text
                prx = True

        if prx and m != "":
            m = n_prx + " " + m

        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(m)["compound"], 3))
    return rule


def rule4(doc):
    """Sentiment modifier + Aspect, a is a child of nsubjpass while m is advmod"""
    rule = []
    for token in doc:
        children = token.children
        a, m = "", ""
        prx = False
        for child in children:
            if (
                child.dep_ == "nsubjpass" or child.dep_ == "nsubj"
            ) and not child.is_stop:
                a = child.text

            if child.dep_ == "advmod" and not child.is_stop:
                m = child.text
                m_sub = child.children
                for child_m in m_sub:
                    if child_m.dep_ == "advmod":
                        m = child_m.text + " " + child.text
                        break

            if child.dep_ == "neg":
                n_prx = child.text
                prx = True

        if prx and m != "":
            m = n_prx + " " + m

        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(m)["compound"], 4))
    return rule


def rule5(doc):
    """Sentiment modifier + Aspect, assuming a is a child of m with nsubj while m is cop"""
    rule = []
    for token in doc:
        children = token.children
        a, buf_var = "", ""
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                a = child.text

            if child.dep_ == "cop" and not child.is_stop:
                buf_var = child.text

        if a != "" and buf_var != "":
            rule.append((a, token.text, sid.polarity_scores(token.text)["compound"], 5))
    return rule


def rule6(doc):
    """Sentiment modifier + Aspect, treating interjections"""
    rule = []
    for token in doc:
        children = token.children
        a, m = "", ""
        if token.pos_ == "INTJ" and not token.is_stop:
            for child in children:
                if child.dep_ == "nsubj" and not child.is_stop:
                    a = child.text
                    m = token.text

        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(m)["compound"], 6))
    return rule


def rule7(doc):
    """Sentiment modifier + Aspect, link between a verb and its complement"""
    rule = []
    for token in doc:
        children = token.children
        a, m = "", ""
        prx = False
        for child in children:
            if child.dep_ == "nsubj" and not child.is_stop:
                a = child.text
            if (child.dep_ == "attr") and not child.is_stop:
                m = child.text

            if child.dep_ == "neg":
                n_prx = child.text
                prx = True

        if prx and m != "":
            m = n_prx + " " + m

        if a != "" and m != "":
            rule.append((a, m, sid.polarity_scores(m)["compound"], 7))
    return rule


def compute_rules(row):
    """Computes all rules from a given review text"""
    reviewer_id = row["reviewerID"]
    hotel_id = row["asin"]
    review_text = clean_nlp(row["reviewText"])
    overall = row["overall"]
    doc = nlp(review_text)

    aspects = []
    aspects.extend(rule1(doc))
    aspects.extend(rule2(doc))
    aspects.extend(rule3(doc))
    aspects.extend(rule4(doc))
    aspects.extend(rule5(doc))
    aspects.extend(rule6(doc))
    aspects.extend(rule7(doc))

    prod_pronouns = ["it", "this", "they", "these"]
    aspects_pairs = [
        (a, m, p, r) if a not in prod_pronouns else ("product", m, p, r)
        for a, m, p, r in aspects
    ]
    return {
        "reviewer_id": reviewer_id,
        "hotel_id": hotel_id,
        "review_text": review_text,
        "aspects": aspects_pairs,
        "overall": overall,
    }


def run(dataset):
    """Computes all rules from a given dataset"""
    return [compute_rules(row) for row in dataset]
