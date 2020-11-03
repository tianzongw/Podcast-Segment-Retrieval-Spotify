import numpy as np
import pandas as pd
import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('wordnet')
import xmltodict


# Simon's  extraction function
def extract_segments(path):
    """Given path to json file containing an episode extracts all segments of that episode,
    including start and end time of each segment."""
    with open(path, "r") as read_file:
        episode = json.load(read_file)
    segments = []
    segment_transcripts = []
    clean_segment_transcripts = []
    # had to do "manual" iteration due to irregularities in data
    iter = 0
    for segment in episode["results"]:
        seg_result = {}

        # make sure there is only one dict in this list (should be true according to dataset description)
        assert len(segment["alternatives"]) == 1
        segment_dict = segment["alternatives"][0]
        # sometimes "alternatives" dict is empty...
        if "words" and "transcript" in segment_dict:
            # add segment number
            seg_result["segNum"] = iter
            # add timestamp of the first word in this segment
            seg_result["startTime"] = segment_dict["words"][0]["startTime"]
            # add timestamp of the last word in this segment
            seg_result["endTime"] = segment_dict["words"][-1]["endTime"]
            # add transcript of this segment
            tr = segment_dict["transcript"]
            seg_result["transcript_clean"] = clean_text(tr)
            seg_result["transcript"] = tr
            segment_transcripts.append(tr)
            clean_segment_transcripts.append(seg_result["transcript_clean"])
            segments.append(seg_result)
            iter += 1

    return {'segments': segments, "full_text": "".join(segment_transcripts),
            "full_text_clean": "".join(clean_segment_transcripts), }


def strip_punctuation(text):
    return text.translate(text.maketrans("", "", string.punctuation))


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    t = text.split(" ")
    res = [lemmatizer.lemmatize(w) for w in t]

    return " ".join(res)


def stem(text):
    stemmer = PorterStemmer()
    t = text.split(" ")
    res = [stemmer.stem(w) for w in t]

    return " ".join(res)


def remove_numbers(text):
    t = re.sub(r'\d+', "", text)

    return t


def clean_text(text):
    t = text.lower()
    t = strip_punctuation(t)
    t = remove_stopwords(t)
    t = remove_numbers(t)
    # t = stem(t)
    t = lemmatize(t)
    return t


def extract_topics(filename):
    with open(filename, "r") as f:
        topics = f.readlines()

    topics = "".join(topics)
    topics = xmltodict.parse(topics)

    temp = {}
    for t in topics['topics']['topic']:
        n = int(t['num'])
        temp[n] = {}
        temp[n]['query'] = t['query']
        #     temp[n]['query_clean'] = clean_text(t['query'])
        temp[n]['type'] = t['type']
        temp[n]['description'] = t['description']
        temp[n]['description_clean'] = clean_text(t['description'])
    return temp


def get_topic_episodes(topics, ix, tf_idf, ix2episode, n_episodes=10):
    topic = topics[ix]["description_clean"]
    # tokens = [x for x in topic.split(" ") if x not in ["", " "]]
    #
    # bigrams = []
    # for gram in nltk.ngrams(tokens, 2):
    #     bigrams.append(" ".join(gram))
    #
    # tokens.extend(bigrams)

    tokens = tokenize(topic)

    score = np.zeros((tf_idf.shape[0]))
    for t in tokens:
        if t in tf_idf.columns:
            score = score + tf_idf.loc[:, t]

    ixs = np.argsort(-score)[0:n_episodes]
    top_episodes = [ix2episode[i] for i in ixs]
    return top_episodes


def tokenize(text, n=2):
    tokens = [x for x in text.split(" ") if x not in ["", " "]]
    grams = []
    for g in range(2, n+1):
        for gram in nltk.ngrams(tokens, g):
            grams.append(" ".join(gram))

    tokens.extend(grams)
    return tokens


def get_tf_idf(training, n=2):
    ix2episode = list(training.keys())
    full_texts_clean = [training[k]['full_text_clean']for k in ix2episode]
    full_texts = [training[k]['full_text']for k in ix2episode]
    vectorizer = TfidfVectorizer(ngram_range=(1, n), max_df=0.9, stop_words={'english'})
    X = vectorizer.fit_transform(full_texts_clean)
    tf_idf_df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names()
    )
    return tf_idf_df, ix2episode, full_texts


def compare_with_qrels(results, qrels, topic):
    df = qrels[qrels.topic == topic]
    for k, v in results.items():
        e = v['episode']
        i = v['interval']
#         print(np.sum(df.episode==e))
        if np.sum(df.episode==e)>0:
            ixs = df.episode == e
            for episode, interval, score in zip(df.episode_start[ixs], df.interval[ixs], df.score[ixs]):
                if is_overlapping(i, interval):
                    print("{} is overlapping with choice in qrels with score {}".format(k, score))


def is_overlapping(x, y):
    return x[0] <= y[1] and y[0] <= x[1]


def get_segment_details(choices, i2segnum, i2episode, i2transcript):
    result = {}
    for ix, i in enumerate(choices):
        result["spotify:episode:{}_{}.0".format(i2episode[i], 30 * i2segnum[i])] = {
            "episode": i2episode[i],
            "start": 30 * i2segnum[i],
            "end": 30 * i2segnum[i] + 120,
            "interval": (30 * i2segnum[i], 30 * i2segnum[i] + 120),
            "transcript": i2transcript[i],
            "rank": ix
        }

    return result


