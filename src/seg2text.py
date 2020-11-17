import json
    
with open('../result/roberta-large-nli-stsb-mean-tokens.json', 'r') as f:
    results = json.load(f)

with open('../data/testing.json', 'r') as f:
    testing_segments = json.load(f)


rslt_text = {}
for topid in results:

    rslt_text[topid] = []
    for result in results[topid]:
        for episode_id in result:
            result_start_time = result[episode_id][0]
            target_episode = testing_segments[episode_id]
            for idx, segment in enumerate(target_episode):
                segment_start_time = float(segment['startTime'][:-1])
                if segment_start_time == result_start_time:
                    prev = target_episode[idx-1]['transcript']
                    nxt = target_episode[idx+1]['transcript']
                    rslt_text[topid].append(prev + segment['transcript'] + nxt)            

print('model: roberta-large-nli-stsb-mean-tokens' )
for key, value in rslt_text.items():
    for seg in value:
        print('Topic ID:', key, 'Extracted Segment:', seg)