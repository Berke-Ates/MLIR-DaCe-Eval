import sys
import os
import pandas as pd

# Remove nussinov

directory = os.fsencode(sys.argv[1])

count = 0
acc = 1

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        df = pd.read_csv(sys.argv[1] + "/" + filename)
        df = df.median()

        sdfg_ms = df.iloc[-1]
        mlir_ms = df.iloc[-2]

        speedup = mlir_ms / sdfg_ms
        acc = acc * speedup
        count = count + 1

        line = '{:<20}  {:<12}'.format(filename + ":", speedup)
        print(line)

        continue
    else:
        continue

geo_mean = acc**(1 / count)
line = '{:<20}  {:<12}'.format("Geometric Mean:", geo_mean)
print(line)
