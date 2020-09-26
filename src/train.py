import json
import xml.etree.ElementTree as ET
import numpy as np

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm



def extract_top_k_segments(topic_embedding, segments_embedding, k = 5):
    cos_scores = []
    for segment_embedding in segments_embedding:
        cos_scores.append(util.pytorch_cos_sim(topic_embedding, segment_embedding)[0])
    
    return np.argpartition(np.array(cos_scores), -k)[-k:]

if __name__ == '__main__':

    # prepare training subset 
    path='../data/training_sub.json'
    with open(path,'r') as f:
        data=json.load(f)

    # prepare topics 
    tree = ET.parse('../data/podcasts_2020_topics_train.xml')
    root = tree.getroot()
    topics=[{'topic-no':element[0].text,'query':element[1].text,'description':element[3].text} for element in root]

    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    # prepare embeddings for all segments 
    # save segments index
    idx_segment_in_episode = []
    segments_embedding = []
    
    # for i in tqdm(range(len(data))):
    for i in tqdm(range(3)):
        episode = data[i]
        # save segment index
        idx_segment_in_episode.append(sum(idx_segment_in_episode) + episode[-1]['segNum'])
        for segment in episode:
            segments_embedding.append(embedder.encode(segment['transcript'], convert_to_tensor=True))

    # prepare embeddings for all topics
    # save top k most similiar segments index
    rslt_idx = []
    for topic in topics:
        topic_embedding = embedder.encode(topic['query'] + topic['description'], convert_to_tensor=True)
        rslt_idx.append(extract_top_k_segments(topic_embedding, segments_embedding))

    #TODO: WE FORGOT TO SAVE EPISODE ID
    # map index back to episode id