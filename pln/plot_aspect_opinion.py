from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import itertools


def compute_cloud_opinion_word(data_aspect, aspect_name):
    """ Display cloud opinion word of positive and negative aspect in data_aspect """
    data_aspect_pos = data_aspect[data_aspect.polarity == 1.0]
    data_aspect_neg = data_aspect[data_aspect.polarity == -1.0]

    text_pos = " ".join(review for review in data_aspect_pos.opinion_word.astype(str))
    text_neg = " ".join(review for review in data_aspect_neg.opinion_word.astype(str))

    stopwords = set(STOPWORDS)

    wordcloud_pos = WordCloud(
        stopwords=stopwords,
        background_color="white",
        width=800,
        height=400,
    ).generate(text_pos)
    wordcloud_neg = WordCloud(
        stopwords=stopwords,
        background_color="white",
        width=800,
        height=400,
    ).generate(text_neg)

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(f"{aspect_name} OPINIONS WORDS")

    ax[0].imshow(wordcloud_pos, interpolation="bilinear")
    ax[0].axis("off")
    ax[0].title.set_text("Positive opinions")

    ax[1].imshow(wordcloud_neg, interpolation="bilinear")
    ax[1].axis("off")
    ax[1].title.set_text("Negative opinions")

    plt.show()


def plot_pie(df_aspect_polarity):
    """ Display a pie plot with total of aspect opinions and with total of positive/negative opinions """
    labels_aspect = list(df_aspect_polarity.index.values)
    sizes_aspect = list(df_aspect_polarity.neg.values + df_aspect_polarity.pos.values)

    colors_polarity = ["black", "white"] * len(labels_aspect)
    sizes_polarity = list(
        itertools.chain(
            *zip(df_aspect_polarity.neg.values, df_aspect_polarity.pos.values)
        )
    )

    w, l, p = plt.pie(
        sizes_aspect,
        labels=labels_aspect,
        startangle=90,
        frame=True,
        textprops={"fontsize": 25},
        autopct="%.2f%%",
        pctdistance=0.85,
        rotatelabels=90,
    )
    [t.set_rotation(320) for t in p]
    plt.pie(
        sizes_polarity,
        colors=colors_polarity,
        radius=0.75,
        startangle=90,
        rotatelabels=25,
    )

    centre_circle = plt.Circle((0, 0), 0.5, color="black", fc="white", linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    fig.set_size_inches(30, 30)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
