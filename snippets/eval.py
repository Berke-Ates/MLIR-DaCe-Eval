import json
import glob
import os

list_of_files = glob.glob('.dacecache/sdfg_0/perf/*json')
latest_file = max(list_of_files, key=os.path.getctime)

f = open(latest_file)
data = json.load(f)

for event in data['traceEvents']:
    print(event['dur'] / (1000))
