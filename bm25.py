import sys
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import rank_bm25 as bm25
import pandas as pd
import numpy as np
import re
import string
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('wordnet')
import xmltodict

from src.utils_stats import *


def main(args):
    # wd = args[0]
    # os.chdir(wd)

    with open(args[0], "r") as f:
        files = f.readlines()

    files = [f.replace("\n", "") for f in files]

    episodes = {}

    for filename in files:
        episodes[os.path.splitext(filename)[0]] = extract_segments(filename)

    tf_idf, ix2episode, full_texts = get_tf_idf(episodes)
    topics = extract_topics(args[1])

    for ti, topic_dict in topics.items():

        print("Topic {} {}: {}".format(ti, topic_dict['query'], topic_dict['description']))
        top_episodes = get_topic_episodes(topics, ti, tf_idf, ix2episode)
        i2episode = [t for t in top_episodes for sd in episodes[t]['segments']]
        i2segnum = [sd['segNum'] for t in top_episodes for sd in episodes[t]['segments']]
        i2transcript = [sd['transcript'] for t in top_episodes for sd in episodes[t]['segments']]
        i2transcript_clean = [sd['transcript_clean'] for t in top_episodes for sd in episodes[t]['segments']]
        i2tokens = [tokenize(x) for x in i2transcript_clean]
        ix = list(range(len(i2segnum)))

        model = bm25.BM25Okapi(i2tokens)

        tokenized_query = tokenize(topic_dict['query'])
        tokenized_query.extend(tokenize(clean_text(topic_dict['description'])))
        choices = model.get_top_n(tokenized_query, ix, 10)

        top10 = get_segment_details(choices, i2segnum, i2episode, i2transcript)

        for k, res in top10.items():
            print("{}: {}".format(res['rank'], res['transcript']))
        print(" ")


if __name__ == "__main__":
    args = sys.argv
    main(args[1::])
