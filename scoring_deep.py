import sys
import os
import rank_bm25 as bm25
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils_stats import *
import pickle


def main(args):

    with open(args[3], 'r') as f:
        pickle.load(f)

    n_gram = 2
    n_episodes = 20
    n_res_segments = 10

    with open(args[0], "r") as f:
        files = f.readlines()

    output_file = args[2]
    results_df = pd.DataFrame(index=np.arnage(0, 9))

    files = [f.replace("\n", "") for f in files]
    episodes = {}

    for filename in files:
        episodes[os.path.splitext(filename)[0]] = extract_segments(filename)

    tf_idf, dct, corpus, ix2episode, full_texts = get_tf_idf_gensim(episodes, n_gram)
    topics = extract_topics(args[1])

    for ti, topic_dict in tqdm(topics.items()):
        print("Topic {} {}: {}".format(ti, topic_dict['query'], topic_dict['description']))

        top_episodes = get_topic_episodes_gensim(
            topics, ti, tf_idf, dct, corpus, ix2episode, n=n_gram, n_episodes=n_episodes
        )
        i2episode = [t for t in top_episodes for sd in episodes[t]['segments']]
        i2segnum = [sd['segNum'] for t in top_episodes for sd in episodes[t]['segments']]
        i2transcript = [sd['transcript'] for t in top_episodes for sd in episodes[t]['segments']]
        i2transcript_clean = [sd['transcript_clean'] for t in top_episodes for sd in episodes[t]['segments']]
        i2tokens = [tokenize(x) for x in i2transcript_clean]
        ix = list(range(len(i2segnum)))

        model = bm25.BM25Okapi(i2tokens)

        tokenized_query = tokenize(topic_dict['query'], n_gram)
        tokenized_query.extend(tokenize(clean_text(topic_dict['description']), n_gram))
        choices = model.get_top_n(tokenized_query, ix, n_res_segments)
        scores = model.get_scores(tokenized_query)

        choice_scores = [scores[x] for x in choices]
        results_df = pd.concat([results_df, pd.DataFrame.from_dict({"topic_{}_bm25".format(ti):choice_scores})],
                               axis=1)

        # topic_res = {'episode_id': ["", "Topic: {}".format(ti)],
        #              'seg_num': ["", "query: {}".format(topic_dict['query'])],
        #              "words": ["", "query: {}".format(topic_dict['description'])]}
        #
        # topic_res['episode_id'].extend([i2episode[i] for i in choices])
        # topic_res['seg_num'].extend([i2segnum[i] for i in choices])
        # topic_res['words'].extend([i2transcript[i] for i in choices])

        # results_df = pd.concat([results_df, pd.DataFrame.from_dict(topic_res)])

        i2transcript_clean = [sd['transcript_clean'] for t in top_episodes for sd in episodes[t]['segments']]
        i2transcript_clean.extend(dl_segs_dict[ti])
        i2tokens = [tokenize(x) for x in i2transcript_clean]

        model = bm25.BM25Okapi(i2tokens)
        scores = model.get_scores(tokenized_query)[-10:-1]
        choice_scores = [scores[x] for x in choices]
        results_df = pd.concat([results_df, pd.DataFrame.from_dict({"topic_{}_dl".format(ti): scores})],
                               axis=1)

        # top = get_segment_details(choices, i2segnum, i2episode, i2transcript)
        #
        # for k, res in top.items():
        #     print("{}: {}".format(res['rank'], res['transcript']))
        # print(" ")

    # results_df.to_csv(output_file, index=False)
    with open(output_file, "w") as f:
        pickle.dump(results_df, f)


if __name__ == "__main__":
    args = sys.argv
    main(args[1::])