import torch
import numpy as np
import xml.etree.ElementTree as ET

from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def extract_topics(topic_file = '../data/podcasts_2020_topics_train.xml'):
    
    """
    collecting topic queries 

    :param topic_file       : file path of the topic file
    :type topic_file        : string
    :return                 : extracted topics
    :rtype                  : dictionary of dictionaries in format { 0:{'query':  , 'description'}, 
                                                                     1:{'query':  , 'description'}, 
                                                                     ...,
                                                                     8:{'query':  , 'description'}, 
                                                                    } 
    """
    tree = ET.parse(topic_file)
    root = tree.getroot()
    topics = {}
    
    for element in root:
        topics[element[0].text] = {
            'query': element[1].text, 'description': element[3].text}

    return topics

def extract_targets(target_file = '../data/podcasts_2020_train.1-8.qrels.txt'):
    
    """
    collecting targets from target file

    :param target_file      : file path of the target file
    :type target_file       : string
    :return                 : extracted targets
    :rtype                  : dictionary in format {topic_id-episode_id: [jump_in_pnt1, jump_in_pnt2, ...],
                                                    topic_id-episode_id: [jump_in_pnt1, jump_in_pnt2, ...],
                                                    ...
                                                    }
    """

    # prepare targets
    with open(target_file, 'r') as f:
        contents = f.readlines()

    # Find topic + episode from training set and extract jump-in point (if rating is better than zero).
    temp_list = [(line[0], line.split()[2].split('_')[0].split(':')[2], line.split()[
                  2].split('_')[1]) for line in contents if line[-2] != '0']

    # Put into dict to make sure we only get one instance of topic + episode. Value is list of jump-in points
    # for episode + topic combination.
    targets = defaultdict(list)
    for line in temp_list:
        targets[line[0]+'-'+line[1]].append(float(line[2]))
    
    return targets

def embed_episodes(data, embedder):
    
    """
    embedding all episodes' segments in given dataset 
 
    :param embedder         : pre-trained model used for embedding
    :type embedder          : SentenceTransformer object
    :param data             : collected episodes to embed
    :type data              : dictionary of list of dictionaries in form {episode_id: [{seg_num:0   ,
                                                                                        start_time: ,
                                                                                        end_time:   ,
                                                                                        transcript}, {}, ..., {}],

                                                                          episode_id: [{seg_num:0   ,
                                                                                        start_time: ,
                                                                                        end_time:   ,
                                                                                        transcript}, {}, ..., {}],
                                                                          ... 
                                                                          }
    :return                 : dictionary of all segment embeddings
    :rtype                  : dictionary in format {episode_id_1:[segment_embedidng_1, segment_embedding_2, ...],
                                                    episode_id_2:[segment_embedidng_1, segment_embedding_2, ...],
                                                    ...
                                                    }
    """
    all_segment_embeddings = {}
    for episode_id in tqdm(data.keys()):
        # prepare segments texts and timespans
        segment_texts = [data[episode_id][i]["transcript"]
                         for i in range(len(data[episode_id]))]
        segment_embeddings = embedder.encode(
            segment_texts, convert_to_tensor=True)
        all_segment_embeddings[episode_id] = segment_embeddings



    return all_segment_embeddings

def find_top_k_segments(topics, data, embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens'), top_k = 5):
    """
    retrive the index of the top_k most similar segments for each topic
 
    :param topics                             : see extract_topics
    :type topics                              : dictionary
    :param all_segment_embeddings             : see embed_episodes
    :type all_segment_embeddings              : dictionary 
    :return                                   : dictionary of all segment embeddings
    :rtype                                    : dictionary in format {
                                                                        1: [{episode_id: time_span_of_segment},{episode_id: time_span_of_segment}, ...],
                                                                        2: [{episode_id: time_span_of_segment},{episode_id: time_span_of_segment}, ...]
                                                                        ...
                                                                        8: [{episode_id: time_span_of_segment},{episode_id: time_span_of_segment}, ...]
                                                                    }
    """
    best_segments = {}
    all_segment_embeddings = embed_episodes(data, embedder)
    for topic_id in topics.keys():
        best_segments[topic_id] = []
        episode_result = {}

        topic_embedding = embedder.encode(
            topics[topic_id]['query'] + ' ' + topics[topic_id]['description'], convert_to_tensor=True)
        # for later index matching
        episode_num = 0

        for episode_id in tqdm(data.keys()):
            # prepare segments texts and timespans
            # segment_texts = [data[episode_id][i]["transcript"]
            #                  for i in range(len(data[episode_id]))]
            segment_timespans = [(float(data[episode_id][i]["startTime"].split('s')[0]), float(
                data[episode_id][i]["endTime"].split('s')[0])) for i in range(len(data[episode_id]))]
            # prepare segment embeddings
            # segment_embeddings=embedder.encode(segment_texts , convert_to_tensor=True)
            # return the index of best matching index
            cos_scores = util.pytorch_cos_sim(
                topic_embedding, all_segment_embeddings[episode_id])[0]
            cos_scores = cos_scores.cpu()
            if len(all_segment_embeddings[episode_id]) < top_k:
                continue
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
        _, indices = torch.topk(
            all_scores, top_k, sorted=False, largest=True)

        for idx in indices:
            idx = idx.item()
            best_segments[topic_id].append(
                {episode_result[idx // top_k][0]: episode_result[idx // top_k][2][idx % top_k]})
        
        return best_segments

def does_overlap(a, b): 
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) > 0

def compute_acc(best_segments, targets):
    """
    compute accuracy of best found segments with given targets

    :param best_segments       : see find_top_k_segments
    :type best_segments        : dictionary
    :param targets             : see extract_targets
    :type targets              : dictionary
    :return                    : None
    """
    n_correct = 0

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