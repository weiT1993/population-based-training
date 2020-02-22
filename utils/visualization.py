import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",size=18)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_color('0.9')

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    # if threshold is not None:
    #     threshold = im.norm(threshold)
    # else:
    #     threshold = im.norm(data.max())/2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text_to_fill = valfmt(data[i, j], None) if data[i, j]!=0 else '-'
            text = im.axes.text(j, i, text_to_fill, fontsize=14, color='white', **kw)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()])
            texts.append(text)

    return texts

def plot_heatmap(data,time_range):
    unique_power = list(np.unique(data[:,0]))
    unique_power.sort()
    unique_freq = list(np.unique(data[:,1]))
    unique_freq.sort()
    # print('unique powers:',unique_power,len(unique_power))
    # print('unique frequencies:',unique_freq,len(unique_freq))
    accuracy_map = {'train':np.zeros(shape=(len(unique_power), len(unique_freq)),dtype=float),
    'test':np.zeros(shape=(len(unique_power), len(unique_freq)),dtype=float)}
    best_test_acc = 0
    for row in data:
        power, freq, train_acc, test_acc = row
        row_idx = unique_power.index(power)
        col_idx = unique_freq.index(freq)
        accuracy_map['train'][row_idx, col_idx] = train_acc
        accuracy_map['test'][row_idx, col_idx] = test_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_power = power
            best_freq = freq
    print('{:d} Best: power {} freq {} test_score = {}'.format(time_range,best_power,best_freq,best_test_acc))

    for data_set in accuracy_map:
        fig, ax = plt.subplots(figsize=(15,15))

        # data =np.ma.masked_where(accuracy_map==0, accuracy_map)

        im, cbar = heatmap(accuracy_map[data_set], unique_power, unique_freq, ax=ax,
                        cmap='YlGn', cbarlabel='%s accuracy, higher is better'%data_set)
        #texts = annotate_heatmap(im, valfmt="{x:.3f}")
        ax.set_xlabel('Frequency',fontsize=18,labelpad=10)
        ax.set_ylabel('Power',fontsize=18,labelpad=10)

        fig.tight_layout()
        plt.savefig('./results/%s_heatmap_%d.png'%(data_set,time_range),dpi=400)
        plt.close()