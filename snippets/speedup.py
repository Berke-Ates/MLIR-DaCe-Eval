import sys
import os
import pandas as pd

directory = os.fsencode(sys.argv[1])

count = 0
acc = 1

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df = pd.read_csv(sys.argv[1] + "/" + filename)
        df = df.median()

        sdfg_ms = df.iloc[-1]
        df = df.iloc[:-1]
        fastest_ms = df.min()

        speedup = fastest_ms / sdfg_ms
        acc = acc * speedup
        count = count + 1

        print(filename, speedup)

        continue
    else:
        continue

geo_mean = acc**(1 / count)
print(geo_mean)
