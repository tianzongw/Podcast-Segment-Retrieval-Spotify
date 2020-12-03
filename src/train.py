import json

from utils import extract_topics, extract_targets, embed_episodes, find_top_k_segments, compute_acc
from sentence_transformers import SentenceTransformer

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context



if __name__ == '__main__':

    topic_file = '../data/podcasts_2020_topics_train.xml'
    target_file = '../data/podcasts_2020_train.1-8.qrels.txt'
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    # embedder = SentenceTransformer('bert-large-nli-mean-tokens')
    # embedder = SentenceTransformer('roberta-base-nli-mean-tokens')
    # embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    # top_k = 1
    top_k = [1,3,5]
    n_samples = [200,500,1000,2000] #TODO: 5,000 in sleep
    
    for n in n_samples:
        for k in top_k:
            datafile = '../data/training_sub_' + str(n) + '.json'
            datafile = '../data/training_sub_' + str(n) + '.json'
            datafile = '../data/training_sub_' + str(n) + '.json'

            

            # prepare training subset
            with open(datafile, 'r') as f:
                data = json.load(f)

            # prepare topics
            topics = extract_topics(topic_file)

            # prepare targets
            targets = extract_targets(target_file)

            # find best matching segments for all queries
            best_segments = find_top_k_segments(topics, data, embedder, k)
            # print(best_segments)
            # print accuracy
            acc = compute_acc(best_segments, targets)

            print('n_samples:', n , 'top_k:', k, 'acc:', acc )

