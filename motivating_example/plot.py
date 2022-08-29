import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def add_median_labels(ax, precision='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] -
                      median.get_xdata()[0]) == 0 else y
        text = ax.text(x,
                       y + 100,
                       f'{value:{precision}}',
                       ha='center',
                       va='bottom',
                       fontweight='bold',
                       color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground=median.get_color()),
            path_effects.Normal(),
        ])


dt = pd.read_csv("timings/timings.csv")
sns.set(style="darkgrid")
sns.set(font_scale=1.5)

plt.figure(figsize=(8, 4))
box_plot = sns.boxplot(data=dt, notch=True)

ax = box_plot.axes
ax.set(ylabel='Runtime [s]')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

add_median_labels(ax)
plt.tight_layout()
plt.savefig('plot.pdf', dpi=300, bbox_inches='tight')
