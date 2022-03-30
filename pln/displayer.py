import numpy as np
import pandas as pd
import itertools
import re
from IPython.core.display import display, HTML

colors = itertools.cycle(
    [
        "cornflowerblue",
        "tomato",
        "forestqreen",
        "blueviolet",
        "mediumvioletred",
        "chocolate",
        "sandybrown",
    ]
)


def extract_aspects(review):
    aspects = []
    for term in review.iterrows():
        aspects.append(term[1]["aspect"])
    return {aspect: color for (aspect, color) in zip(np.unique(aspects), colors)}


def count_polarity(review, aspects):
    return {
        aspect: {
            "polarity": np.mean(
                [
                    t[1]["polarity"]
                    for t in review.iterrows()
                    if aspect in t[1]["aspect"]
                ]
            ),
            "color": color,
        }
        for (aspect, color) in aspects.items()
    }


def parse_colors(review, aspects):
    p_colors = []
    prev_end = -1
    for term in review.iterrows():
        aspect = term[1]["aspect"]
        phrase = term[1]["aspect_term"]
        if aspect not in aspects:
            continue
        start = term[1]["text"].find(phrase)
        if start == -1:
            continue
        end = start + len(phrase)
        if start <= prev_end:
            start = prev_end
        if start >= end:
            continue
        prev_end = end
        p_colors.append(
            (start, end, term[1]["polarity"], aspects[aspect], term[1]["opinion_word"])
        )
    return p_colors


def underline(text, color, score):
    if score == 0.0:
        bg = "#92C5F0"
    elif score == 1.0:
        bg = "#BFF0C0"
    else:
        bg = "#F09892"
    return (
        f'<u style="text-decoration: underline dotted;text-decoration-color: {color}; text-decoration-thickness:5px; background-color: {bg}"">'
        + text
        + "</u>"
    )


def bold_adjs(review, adjs):
    for adj in adjs:
        review = re.sub(r"(" + adj + ")", r"<b>\1</b>", review, flags=re.IGNORECASE)
    return review


def draw_review(review, color_map):
    if len(color_map) < 1:
        return review
    text = review[: color_map[0][0]]
    adjs = []
    for i, (start, end, score, color, adj) in enumerate(color_map):
        t = review[start:end]
        text += underline(t, color, score)
        if len(color_map) != i + 1:
            text += review[end : color_map[i + 1][0]]
        else:
            text += review[end:]
        adjs.append(adj)
    return text, adjs


def html_scores(aspects_score):
    html_list = ""
    for aspect, data in aspects_score.items():
        html_list += f'<tr><td> <p style="text-decoration:underline dotted;text-decoration-color: {data["color"]}; text-decoration-thickness:5px;">{aspect}</p></td><td>{data["polarity"]:.1f}</td></tr>\n'
    return f"""
    <h4> Polarity </h4>
    <table>
    {html_list}
    </table>
    """


def create_html(review, aspects_score, color_map, review_id):
    text, adjs = draw_review(review["text"][0], color_map)
    return f"""
    <html>
    <h1>Review {review_id} </h1>
    <center>{html_scores(aspects_score)}</center>
    <br></br>
    <div style="font-size: large">{bold_adjs(text, adjs)}</div>
    </html>
    """


def review_to_html(review, review_id):
    aspects = extract_aspects(review)
    aspects_score = count_polarity(review, aspects)
    color_map = parse_colors(review, aspects)
    return create_html(review, aspects_score, color_map, review_id)


def run_display(review_id, review):
    display(HTML(review_to_html(review, review_id)))
