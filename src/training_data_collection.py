import os
import json
# Dataset Structure
'''
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

'''
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

def collect_training_episodes(root_dir, input_file):
    episodes = []
    training_episodes = []

    with open(input_file) as f:
        for cnt, line in enumerate(f):
            if not line.split()[-1] == '0': #remove non-relevant episodes
                episode = line.split()[2].split(':')[-1].split('_')[0] + '.json'
                if episode not in episodes:
                        episodes.append(episode)

    for (root, dirs, files) in os.walk(root_dir, topdown=True): 
        for file in files:
            if file in episodes:
                training_episodes.append(root+'/'+file)
    
    return training_episodes

if __name__ == '__main__':

    input_file = '../data/podcasts_2020_train.1-8.qrels.txt'
    root_dir = '../data/podcasts-no-audio-13GB'
    training_episodes = collect_training_episodes(root_dir, input_file)
    training_segments = {}
    for episode in training_episodes:
        episode_id=episode.split('/')[-1].split('.json')[0]
        training_segments[episode_id]=extract_segments(episode)
    with open('../data/training_sub.json', 'w') as fout:
            json.dump(training_segments , fout)


