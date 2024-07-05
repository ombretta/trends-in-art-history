import pandas as pd
import matplotlib.pyplot as plt

def plot_frequency(data, title, xlabel, ylabel, filename, bar_colors=None, log_scale=False,
                   figsize=(10,6), fontsize=20, orientation='horizontal', legend_handles=None, legend_labels=None):
    plt.figure(figsize=figsize)

    if orientation == 'horizontal':
        bars = plt.barh(data.index, data.values, color=bar_colors)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        if log_scale:
            plt.xscale('log')
    elif orientation == 'vertical':
        bars = plt.bar(data.index, data.values, color=bar_colors)
        plt.xlabel(ylabel, fontsize=fontsize)
        plt.ylabel(xlabel, fontsize=fontsize)
        plt.xticks(rotation=90, ha='right', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if log_scale:
            plt.yscale('log')

    # Highlight the bars for the selected artists
    if bar_colors:
        for bar, color in zip(bars, bar_colors):
            bar.set_color(color)

    # Create a legend for the styles
    if legend_handles and legend_labels:
        plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(filename, transparent=True, bbox_inches='tight')
    plt.show()


def plot_frequency_with_annotations(data, title, xlabel, ylabel, filename, bar_colors=None, log_scale=False,
                   figsize=(10,6), fontsize=20, orientation='horizontal', legend_handles=None, legend_labels=None,
                   x_annotations=None, y_annotations=None, annotations_labels=None, label_x_offset=None,
                   label_y_offset=None):

    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)

    bars = plt.bar(data.index, data.values, color=bar_colors)
    plt.xlabel(ylabel, fontsize=fontsize)
    plt.ylabel(xlabel, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim(left=-1, right=len(data))
    plt.xticks([])  # Remove xticks
    if log_scale:
        plt.yscale('log')

    # Highlight the bars for the selected artists
    if bar_colors:
        for bar, color in zip(bars, bar_colors):
            bar.set_color(color)

    if annotations_labels:
        for i, label in enumerate(annotations_labels):
            ax.annotate(label,
                        xy=(x_annotations[i], y_annotations[i]+1), xycoords='data',
                        xytext=(x_annotations[i]+label_x_offset[i], y_annotations[i]+label_y_offset[i]), textcoords='data',
                        arrowprops=dict(arrowstyle="simple", connectionstyle="angle3,angleA=0,angleB=90",
                                        facecolor='white'), fontsize=fontsize-20)

    # Create a legend for the styles
    if legend_handles and legend_labels:
        plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize)

    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close()


def main():

    # Set Times New Roman as the default font
    plt.rcParams['font.family'] = 'Times New Roman'

    # Load prediction data
    split = "test"
    results_dir = "highest_expl_variance/"
    test_predictions_file = results_dir + split + "_prediction.csv"
    style_pred = pd.read_csv(test_predictions_file)

    # Define the order of styles
    styles_order = [
        "early renaissance", "northern renaissance", "high renaissance", "mannerism late renaissance", "baroque",
        "rococo", "romanticism", "ukiyo e", "realism", "impressionism", "symbolism", "post impressionism",
        "pointillism", "art nouveau modern", "fauvism", "analytical cubism", "expressionism", "synthetic cubism",
        "cubism", "naive art primitivism", "abstract expressionism", "action painting", "color field painting",
        "new realism", "pop art", "minimalism", "contemporary realism"
    ]

    # List of specified artists to highlight
    selected_artists = [
        'Frans Hals', 'Fra Angelico', 'Claude Monet', 'Albrecht Durer', 'Andy Warhol',
        'Egon Schiele', 'Juan Gris', 'Pablo Picasso', 'Paul Cezanne', 'Henri Matisse',
        'Salvador Dali', 'Gustave Courbet'
    ]

    # Extract artist name from the image path
    style_pred['artist'] = style_pred['images'].apply(lambda x: x.split('/')[1].rsplit('_', 1)[0])

    # Check which artists have multiple styles
    painters_styles = style_pred.groupby('artist')['gt'].agg(pd.Series.mode)
    # multiple_styles = painters_styles[painters_styles.apply(lambda x: isinstance(x, pd.Series) and len(x) > 1)]
    # multiple_styles.to_csv(results_dir + 'artists_with_multiple_styles_' + split + '_set.csv', index=True)

    # Check if selected artists exist in the data
    for artist in selected_artists:
        print(artist, any(style_pred['artist'].str.contains(artist.replace(" ", "-").lower())))

    # Plot artist frequency in the test results
    n = 100
    artist_frequency = style_pred["artist"].value_counts()

    # Create a color map for styles
    color_map = {value: plt.cm.plasma(i / len(styles_order)) for i, value in enumerate(styles_order)}

    # Map colors to styles in the painter grouped data
    bar_colors = [color_map.get(style_pred[style_pred['artist'] == artist]['gt'].mode().values[0], 'grey')
                  for artist in artist_frequency.index]
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[value], linestyle='', markersize=10)
               for value in styles_order]

    # # Make plot of painter frequency
    plot_frequency(artist_frequency[:n], 'Artist Frequency', 'Count', 'Artist',
                   results_dir + '/' + str(n) + '_artist_frequency_' + split + '.pdf',
                   bar_colors=bar_colors[:n], log_scale=True, figsize=(80, 20), fontsize=25, orientation="vertical",
                   legend_handles=handles, legend_labels=styles_order)

    # Make plot of style frequency
    style_frequency = style_pred["gt"].value_counts()
    bar_colors = [color_map.get(style, 'grey') for style in style_frequency.index]
    plot_frequency(style_frequency, 'Style Frequency', 'Count', 'Artist',
                   results_dir + '/style_frequency_' + split + '.pdf',
                   bar_colors=bar_colors, log_scale=False, figsize=(30, 15), fontsize=25, orientation="vertical",
                   legend_handles=handles, legend_labels=styles_order)

    # Classification accuracy
    # Filter out painters with less than n paintings
    n_paintings = 50
    filtered_artists = artist_frequency[artist_frequency > n_paintings].keys().tolist()
    filtered = style_pred[style_pred['artist'].isin(filtered_artists)].copy()
    filtered['correct'] = (filtered['gt'] == filtered['preds']).astype(int)
    accuracy = filtered.groupby("artist")['correct'].mean()*100
    pd.DataFrame({"artist": accuracy.index, "accuracy": accuracy.values, "n_paintings":
        filtered.groupby("artist")['correct'].count().values}).to_csv(results_dir + "/accuracy_per_artist.csv")
    sorted_accuracy = accuracy.sort_values(ascending=False)
    bar_colors = [color_map.get(filtered[filtered['artist'] == artist]['gt'].mode().values[0], 'grey')
                  for artist in sorted_accuracy.index]



    x_annotations = [sorted_accuracy.index.get_loc(artist.replace(" ", "-").lower()) for artist in selected_artists]
    y_annotations = [sorted_accuracy[artist.replace(" ", "-").lower()] for artist in selected_artists]

    artists_labels = [
        'Frans Hals',
        'Fra Angelico',
        'Claude Monet',
        'Albrecht Dürer',
        'Andy Warhol',
        'Egon Schiele',
        'Juan Gris',
        'Pablo Picasso',
        'Paul Cézanne',
        'Henri Matisse',
        'Salvador Dalí',
        'Gustave Courbet'
    ]

    label_x_offset = [10, 8, 10, 13, 9, 2, 6, -13, -5, 6, -10, -12]
    label_y_offset = [3, -3, -2, 7, 14, 9, 7, 30, 35, 30, 35, 15]

    plot_frequency_with_annotations(sorted_accuracy, None, 'Accuracy (%)', 'Artist',
                   results_dir + '/per_artist_accuracy_' + split + '_set.pdf',
                   bar_colors=bar_colors, log_scale=False, figsize=(50, 15), fontsize=80, orientation="vertical",
                   x_annotations=x_annotations, y_annotations=y_annotations, annotations_labels=artists_labels,
                   label_x_offset=label_x_offset, label_y_offset=label_y_offset)


    return style_pred, painters_styles


if __name__ == '__main__':
    style_pred, painters_styles = main()
