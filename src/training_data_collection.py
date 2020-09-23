import os

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

def collect_training_episodes(root_dir, input_file):
    episodes = []
    training_episodes = []

    with open(input_file) as f:
        for cnt, line in enumerate(f):
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

    for episode in training_episodes:
        print(episode)


