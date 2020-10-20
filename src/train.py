import json

from utils import extract_topics, extract_targets, embed_episodes, find_top_k_segments, compute_acc
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':


    datafile = '../data/training_sub.json'
    topic_file = '../data/podcasts_2020_topics_train.xml'
    target_file = '../data/podcasts_2020_train.1-8.qrels.txt'
    embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    top_k = 5

    # prepare training subset
    with open(datafile, 'r') as f:
        data = json.load(f)

    # prepare topics
    topics = extract_topics(topic_file)

    # prepare targets
    targets = extract_targets(target_file)

    # find best matching segments for all queries
    best_segments = find_top_k_segments(topics, data, embedder, top_k)
    
    # print accuracy
    compute_acc(best_segments, targets)

