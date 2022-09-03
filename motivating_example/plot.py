import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def add_median_labels(ax, precision='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] -
                      median.get_xdata()[0]) == 0 else y
        text = ax.text(x,
                       y + 75,
                       f'{value:{precision}}',
                       ha='center',
                       va='bottom',
                       fontweight='bold',
                       color='black',
                       fontfamily='Arial')
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=2, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])


dt = pd.read_csv("timings/timings_clang.csv")
sns.set(style="darkgrid")
sns.set(font_scale=1.25)

plt.figure(figsize=(8, 6))
box_plot = sns.boxplot(data=dt, notch=True)

ax = box_plot.axes
ax.set(ylabel='Runtime [ms]')
ax.set_xticklabels(["GCC", "Clang", "DaCe", "Polygeist\n+ MLIR", "DCIR"])

add_median_labels(ax)
plt.tight_layout()
plt.savefig('plot.pdf', dpi=300, bbox_inches='tight')
