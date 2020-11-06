import os
import json
import random
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='args for data collection')

parser.add_argument('--r', '--noise_ratio', default=0.0001, type=float, help='noise ratio for training experiments')
parser.add_argument('--mode',  default='train')
args = parser.parse_args()

# Dataset Structure
"""
total num of episodes     : 125642  (100%)
labeled episodes          : 344     (0.3%)
labeled relevant episodes : 100     (0.1%)    

├── podcasts-no-audio-13GB
│   ├── 150gold.egfb.txt
│   ├── 150gold.tsv
│   ├── README.txt
│   ├── metadata.tsv
│   ├── podcast-dataset-description-paper.pdf
│   ├── podcasts-test
│   │   ├── metadata-summarization-testset.tsv
│   │   ├── podcasts-transcripts-summarization-testset
│   │   │   ├── 0
│   │   │   │   ├── 1
│   │   │   │   │   ├── show_015DbLwcXu2fK7e9jIfbFo
│   │   │   │   │   │   └── 74t5WREXUbhEKNI89CNSkL.json
│   │   │   │   │   ├── show_01DbRiALDPdvZdoiY8yQL6
│   │   │   │   │   │   └── 5fG4VlWnWwzAt6mSs0H7lY.json

                    ...


│   │   │   ├── 1
│   │   │   │   ├── 0
│   │   │   │   │   ├── show_10EdetMrUehkg7Cfk1c6IS
│   │   │   │   │   │   └── 3L9UtDN1bPoKcEGU1BmCeT.json
│   │   │   │   │   ├── show_10F8QlDNqpbH8RcQ6gEM8U
│   │   │   │   │   │   └── 0UoTnPStSJhNpGGrEqomO7.json

                    ...

"""
def extract_segments(path):
    with open(path, "r") as read_file:
        episode = json.load(read_file)
    segments=[]
    #had to do "manual" iteration due to irregularities in data
    iter=0
    for segment in episode["results"]:
        seg_result={}
        #make sure there is only one dict in this list (should be true according to dataset description)
        assert len(segment["alternatives"])==1
        segment_dict=segment["alternatives"][0]
        #sometimes "alternatives" dict is empty...
        if "words" and "transcript"  in segment_dict:
            #add segment number
            seg_result["segNum"]=iter
            #add timestamp of the first word in this segment
            seg_result["startTime"]=segment_dict["words"][0]["startTime"]
            #add timestamp of the last word in this segment
            seg_result["endTime"]=segment_dict["words"][-1]["endTime"]
            #add transcript of this segment 
            seg_result["transcript"]=segment_dict["transcript"]
            segments.append(seg_result)
            iter+=1

    return segments

def collect_episodes(root_dir, labeled_file, ratio = 0, matching_level = [1,2,3,4]):
    
    """
    collecting episodes as searching space at different noise_ratio and matching level

    :param ratio            : percent of searching space among all episodes
    :param matching_level   : relevant episodes at different matching level
    :type a                 : float between 0-1, default 0
    :type b                 : list of a combination of int from 1-4 
    :return                 : list of episodes path 
    :rtype                  : list 
    """

    episodes = []
    training_episodes = []
    
    # collrct 100 relevant episodes
    with open(labeled_file) as f:
        for _, line in enumerate(f):
            if int(line.split()[-1]) in matching_level: # filtering the level we want to match, 
                episode = line.split()[2].split(':')[-1].split('_')[0] + '.json'
                if episode not in episodes:
                        episodes.append(episode)

    for (root, _, files) in os.walk(root_dir, topdown=True): 

        for file in files:
            if file in episodes or random.uniform(0, 1) <= ratio:
                if 'xml' not in root+'/'+file:
                    training_episodes.append(root+'/'+file)

    return training_episodes

if __name__ == '__main__':

    if args.mode == 'train':
        labeled_file = '../data/podcasts_2020_train.1-8.qrels.txt'
        root_dir = '../data/podcasts-no-audio-13GB'
        training_episodes = collect_episodes(root_dir, labeled_file, ratio = args.r)
        training_segments = {}
        for episode in training_episodes:
            episode_id=episode.split('/')[-1].split('.json')[0]
            training_segments[episode_id]=extract_segments(episode)
        with open('../data/training_sub.json', 'w') as fout:
                json.dump(training_segments , fout)

    if args.mode == 'test':
        input_file = '../data/10k_testset.csv'
        root_dir = '../data/podcasts-no-audio-13GB'

        episodes = pd.read_csv(input_file)['episode_uri'].to_list()

        test_segments = {}
        count = 0
        for (root, _, files) in os.walk(root_dir, topdown=True): 

            for file in files:
                if file.split('.')[0] in episodes:
                    test_segments[file.split('.')[0]] = extract_segments(root+'/'+file)
        

        with open('../data/testing.json', 'w') as fout:
            json.dump(test_segments , fout)




