import json
import xml.etree.ElementTree as ET
import numpy as np
import torch

from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


if __name__ == '__main__':

    # prepare training subset
    path = '../data/training_sub.json'
    with open(path, 'r') as f:
        data = json.load(f)
    print(len(data))
    # prepare topics
    tree = ET.parse('../data/podcasts_2020_topics_train.xml')
    root = tree.getroot()
    topics = {}
    for element in root:
        topics[element[0].text] = {
            'query': element[1].text, 'description': element[3].text}

    # prepare targets
    with open('../data/podcasts_2020_train.1-8.qrels.txt', 'r') as f:
        contents = f.readlines()

    # Find topic + episode from training set and extract jump-in point (if rating is better than zero).
    temp_list = [(line[0], line.split()[2].split('_')[0].split(':')[2], line.split()[
                  2].split('_')[1]) for line in contents if line[-2] != '0']

    # Put into dict to make sure we only get one instance of topic + episode. Value is list of jump-in points
    # for episode + topic combination.
    targets = defaultdict(list)
    for line in temp_list:
        targets[line[0]+'-'+line[1]].append(float(line[2]))

    # print(targets)
    # prepare embedder
    #embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    '''
    final result looks smth like
    {
        1: {episode_id: time_span_of_segment},
        2: {episode_id: time_span_of_segment},
        ...
        8: {episode_id: time_span_of_segment}
    }
    '''

    all_segment_embeddings = {}
    for episode_id in tqdm(data.keys()):
        # prepare segments texts and timespans
        segment_texts = [data[episode_id][i]["transcript"]
                         for i in range(len(data[episode_id]))]
        segment_embeddings = embedder.encode(
            segment_texts, convert_to_tensor=True)
        all_segment_embeddings[episode_id] = segment_embeddings

    best_segments = {}

    top_k = 5

    # find best matching segments for all queries
    for topic_id in topics.keys():
        best_segments[topic_id] = []
        episode_result = {}

        topic_embedding = embedder.encode(
            topics[topic_id]['query'] + ' ' + topics[topic_id]['description'], convert_to_tensor=True)
        # for later index matching
        episode_num = 0

        for episode_id in tqdm(data.keys()):
            # prepare segments texts and timespans
            segment_texts = [data[episode_id][i]["transcript"]
                             for i in range(len(data[episode_id]))]
            segment_timespans = [(float(data[episode_id][i]["startTime"].split('s')[0]), float(
                data[episode_id][i]["endTime"].split('s')[0])) for i in range(len(data[episode_id]))]
            # prepare segment embeddings
            # segment_embeddings=embedder.encode(segment_texts , convert_to_tensor=True)
            # return the index of best matching index
            cos_scores = util.pytorch_cos_sim(
                topic_embedding, all_segment_embeddings[episode_id])[0]
            cos_scores = cos_scores.cpu()
            best_segment_idx = np.argpartition(-cos_scores,
                                               range(top_k))[0:top_k]

            # TODO: IF NEED TO EXTRACT TEXT ADD [segment_texts[idx] for idx in best_segment_idx] AS WELL
            episode_result[episode_num] = (episode_id, cos_scores[best_segment_idx], [
                                           segment_timespans[idx] for idx in best_segment_idx])
            episode_num += 1

        all_scores = []
        for episode_id in episode_result.keys():
            all_scores.append(episode_result[episode_id][1])

        all_scores = torch.stack(all_scores).reshape(-1)
        values, indices = torch.topk(
            all_scores, top_k, sorted=False, largest=True)

        for idx in indices:
            idx = idx.item()
            best_segments[topic_id].append(
                {episode_result[idx // top_k][0]: episode_result[idx // top_k][2][idx % top_k]})

    # compute accuracy
    def does_overlap(a, b): return max(
        0, min(a[1], b[1]) - max(a[0], b[0])) > 0
    n_correct = 0

    print(best_segments)
    for topic_id in best_segments.keys():
        for i in range(len(best_segments[topic_id])):
            n_correct_element = 0
            for episode_id in best_segments[topic_id][i].keys():
                if any([does_overlap(best_segments[topic_id][i][episode_id], (target_timespan, target_timespan+120)) for target_timespan in targets[topic_id + '-' + episode_id]]):
                    n_correct_element += 1

            if n_correct_element > 0:
                n_correct += 1
                print(topic_id)
                break

    print('acc: ', n_correct/len(best_segments))

    # TODO: Failed on topic 6, check
    # TODO: Modulize code...
