from math import ceil
import sys
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt = []
num_bench = len(sys.argv) - 2
rows = 3
cols = ceil(num_bench / rows)
color = ['green', 'green', 'green', 'red']

sns.set(style="darkgrid")
# sns.set(font_scale=1.2)

fig, ax = plt.subplots(rows, cols, sharex='row')
fig.set_figheight(10)
fig.set_figwidth(15)
fig.supylabel('Runtime [s]')

for i in range(num_bench):
    dt.append(pd.read_csv(sys.argv[i + 2]))

for i in range(rows):
    for j in range(cols):
        if (i * cols + j >= num_bench):
            ax[i, j].axis('off')
            break

        sns.barplot(data=dt[i * cols + j],
                    palette=color,
                    estimator=np.median,
                    ax=ax[i, j])
        bench_name = os.path.splitext(
            os.path.basename(sys.argv[i * cols + j + 2]))[0]
        ax[i, j].set_title(bench_name)
        ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(),
                                 rotation=90,
                                 ha="center")
        if (i < rows - 1):
            ax[i, j].set(xticklabels=[])

plt.tight_layout()
plt.savefig(sys.argv[1], dpi=300, bbox_inches='tight')
