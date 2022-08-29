import sys
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
                       y + 300,
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


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


dt = pd.read_csv(sys.argv[1])
sns.set(style="darkgrid")
sns.set(font_scale=1.5)

plt.figure(figsize=(8, 4))
box_plot = sns.boxplot(data=dt, notch=True)

ax = box_plot.axes
# ax.set_title(sys.argv[3])
ax.set(ylabel='Runtime [ms]')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

add_median_labels(ax)
# change_width(ax, .5)

plt.tight_layout()
plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
