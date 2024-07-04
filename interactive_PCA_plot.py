# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:15:31 2024

@author: gestaltrevision
"""

import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import plotly.express as px
import plotly.graph_objects as go

import torch
from torch.utils.data import Dataset

styles_order = {
    "WikiArt": [
        'Early_Renaissance', 'Northern_Renaissance', 'High_Renaissance', 'Mannerism_Late_Renaissance',
        'Baroque', 'Rococo', 'Romanticism', 'Ukiyo_e', 'Realism', "Impressionism", "Pointillism",
        "Post_Impressionism", "Symbolism", "Art_Nouveau_Modern", "Fauvism", 'Analytical_Cubism', "Cubism",
        "Synthetic_Cubism", "Expressionism", 'Naive_Art_Primitivism', "Abstract_Expressionism", "Action_painting",
        "Color_Field_Painting", "New_Realism", 'Pop_Art', 'Minimalism', 'Contemporary_Realism'
    ],
}


class ArtDataset(Dataset):
    def __init__(self, annotations_file, img_dir, attribute='style'):
        df = pd.read_csv(annotations_file).sort_values(attribute).reset_index(drop=True)
        self.labels, self.class_names = pd.factorize(df[attribute])
        self.img_dir = img_dir
        self.img_files = df['file']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx].split("/")[1])
        image = torch.load(img_path.replace(".jpg", ".pt"), map_location=torch.device(self.device)).squeeze(0)
        label = self.labels[idx]
        return image.to(torch.float32), label

    def get_class_name(self, idx):
        return self.class_names[idx]


def inference(model, loader, img_files, device, savedir):
    model.eval()

    all_labels = []
    all_predictions = []

    latent_dim = model.model[-1].in_features
    latent_features = np.zeros([len(loader), latent_dim])

    with torch.no_grad():
        for i, (data, img_path) in enumerate(zip(loader, img_files)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs, latent = model.inference(images)

            logits = torch.nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            style_dir = img_path.split('/')[0]
            if not os.path.exists(savedir + style_dir):
                os.mkdir(savedir + style_dir)
            np.save(savedir + img_path.replace(".jpg", ".npy"), latent.numpy())

            latent_features[i] = latent.numpy()[0, :]

    accuracy = accuracy_score(all_labels, all_predictions)  # Normalized accuracy

    return accuracy, latent_features


def pca(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("Explained variance ratio", pca.explained_variance_ratio_)
    # Projection in the principal components
    return pca.fit_transform(X), [round(e, 2) for e in pca.explained_variance_ratio_]


def get_border_points(X_red):
    hull = ConvexHull(X_red)
    border_points = X_red[hull.vertices]
    return border_points


def from_coordinated_to_filename(xy, X_red, filenames):
    '''
    :param xy: 1d array [x, y] containing point coordinates. Can be extended to 3-d as well.
    :param X_red: the np array containing the ordered features vectors reduced with PCA.
    :param filenames: List of ordered filenames in the dataset.
    :return: Filename of the corresponding painting.
    '''
    target = np.array(xy)
    matches = np.all(X_red == target, axis=1)
    ind = np.where(matches)[0][0]
    painting = filenames[ind]
    return painting


def from_filename_to_coordinates(painting_filename, X_red, filenames):
    '''

    :param painting_filename: Filename of the painting to find in the plot. This needs to be
    precisely the same text as the annotations.
    :param X_red: the np array containing the ordered features vectors reduced with PCA.
    :param filenames: List of ordered filenames in the dataset.
    :return: 1d array [x, y] containing point coordinates. Can be extended to 3 coordinates as well.
    '''

    # Painting not found
    if len(filenames[filenames == painting_filename]) == 0:
        print(painting_filename, 'not found.')
        return None
    ind = filenames[filenames == painting_filename].index[0]
    xy = X_red[ind][:2]
    return xy


def get_painters_coordinates(X_red, filenames, filter_freq=25):
    painters_check = []
    painters = []
    for filename in filenames.values:
        painter = filename.split("/")[1].split("_")[0]
        print(painter)
        if painter not in painters_check:
            painters_check.append(painter)
            painters.append(filename)
    print("painters", len(painters_check))
    print("painters", len(painters))

    coordinates = []
    filtered_painters = []
    for i, painting_filename in enumerate(painters):
        freq = len(filenames[filenames.str.contains(painters_check[i])])
        # print(painters_check[i], freq)

        if freq >= filter_freq:
            filtered_painters.append(painters_check[i])
            xy = from_filename_to_coordinates(painting_filename, X_red, filenames)
            coordinates.append(xy)

    return filtered_painters, coordinates


def pca2d(X, labels, styles, filenames, ordered_styles, split, savedir, dataset_name="LAPIS", n_components=2,
          fontsize=20):
    print("PCA -", split, "set")

    X_red, explained_variance = pca(X, n_components)
    border_points = get_border_points(X_red)

    # Get the paintings filenames of the border points
    border_paintings = []
    for point in border_points:
        painting = from_coordinated_to_filename(point, X_red, filenames)
        border_paintings.append(painting)

    c = [ordered_styles.index(styles[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(17, 12))

    scatter = ax.scatter(X_red[:, 0], X_red[:, 1], c=c, label=ordered_styles, cmap='plasma')

    # Highlight border points
    for i, point in enumerate(border_points):
        ax.scatter(point[0], point[1], s=160, edgecolor='k', facecolor='none')

        ax.text(point[0], point[1], f'({border_paintings[i]})', fontsize=fontsize, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Axis labels
    ax.set_title(dataset_name + " - " + split + " (Expl. variance " + str(explained_variance) + ")", fontsize=fontsize)
    ax.set_xlabel("PCA 1", fontsize=fontsize)
    ax.set_ylabel("PCA 2", fontsize=fontsize)

    ax = add_legend(ax, ordered_styles, labels, styles)

    # Display the plot
    plt.savefig(savedir + split + "_pca1_2.png", bbox_inches='tight')

    return X_red


def pca3d(X, labels, styles, ordered_styles, split, savedir, dataset_name="LAPIS", n_components=3):
    print("PCA -", split, "set")

    X_red, explained_variance = pca(X, n_components)

    fig = plt.figure(figsize=(17, 12))

    # Create a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_red[:, 0], X_red[:, 1], X_red[:, 2], c=labels, cmap='plasma')

    # Axis labels
    ax.set_title(dataset_name + " - " + split + " (Expl. variance " + str(explained_variance) + ")", fontsize=25)
    ax.set_xlabel("PCA 1", fontsize=20)
    ax.set_ylabel("PCA 2", fontsize=20)
    ax.set_zlabel("PCA 3", fontsize=20)

    ax = add_legend(ax, ordered_styles, labels, styles)

    # Display the plot
    plt.savefig(savedir + split + "_pca1_2_3.png", bbox_inches='tight')


def add_legend(ax, ordered_styles, labels, label_names):
    # Sort label_names according to styles_order

    # print("label_names", label_names)
    ordered_label_names = sorted(label_names, key=lambda x: ordered_styles.index(x))

    # Create a custom legend
    legend_labels = [ordered_label_names[i] for i in np.unique(labels)]  # Get labels corresponding to unique colors
    colors = [plt.cm.plasma(i / len(label_names)) for i in np.unique(labels)]  # Get color for each unique class

    # Generate custom legend handles
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]

    # Add the custom legend to the plot, outside the plot area
    ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.05, 1.05), title="Styles", fontsize=18,
              title_fontsize=18)

    # Adjust the subplot to fit the legend outside the plot area
    plt.subplots_adjust(right=0.75)

    return ax


def pca2d_interactive(X, labels, styles, filenames, ordered_styles, split, savedir, dataset_name="LAPIS",
                      n_components=2):
    print("PCA -", split, "set")

    # Perform PCA
    X_red, explained_variance = pca(X, n_components)

    print("ordered_styles", ordered_styles)
    print("labels", labels)
    print("styles", styles)

    # Colors
    c = [ordered_styles.index(styles[l]) for l in labels]

    # Prepare data for Plotly
    df = pd.DataFrame(X_red, columns=['PCA 1', 'PCA 2'])
    df['Label'] = labels
    # df['Style'] = [ordered_styles.index(styles[l]) for l in labels]
    df['Style'] = [f.split("/")[0] for f in filenames]
    df['Artwork'] = [f.split("_")[-1].split(".jpg")[0] for f in filenames]
    df['Artist'] = [f.split("/")[1].split("_")[0] for f in filenames]

    # Create a hover text column
    df['Hover'] = df.apply(lambda row: f"Artwork: {row['Artwork']}<br>Artist: {row['Artist']}<br>Style: {row['Style']}",
                           axis=1)

    # Create a scatter plot
    fig = px.scatter(df, x='PCA 1', y='PCA 2', color=c, hover_name='Label',
                     hover_data={'Hover': False, 'Style': True, 'Artwork': True,
                                 'Artist': True, 'PCA 1': False, 'PCA 2': False, "Label": False})

    # # Prepare data for Plotly
    df = pd.DataFrame(X_red, columns=['PCA 1', 'PCA 2'])
    df['Label'] = labels
    df['Style'] = [ordered_styles.index(styles[l]) for l in labels]
    df['Filename'] = filenames

    # Create a hover text column
    df['Hover'] = df.apply(lambda row: f"Filename: {row['Filename']}<br>Style: {row['Style']}", axis=1)

    # Create a scatter plot
    fig = px.scatter(df, x='PCA 1', y='PCA 2', color='Style', hover_name='Label',
                     hover_data={'Hover': True, 'Style': False, 'Filename': False})

    # Customize the layout
    fig.update_layout(
        title=f"{dataset_name} - {split} (Expl. variance {explained_variance})",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        legend_title="Style",
        font=dict(size=18),
        width=1000,
        height=800
    )

    # Save the plot as an HTML file
    fig.write_html(f"{savedir}{split}_pca1_2_interactive.html")

    # Show the plot
    fig.show()

    return X_red


def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def main(dataset_name, annotations_dir, save_dir, attribute):
    ordered_styles = styles_order[dataset_name]

    train_dataset = ArtDataset(os.path.join(annotations_dir, dataset_name + "_train.csv"), "", attribute)
    test_dataset = ArtDataset(os.path.join(annotations_dir, dataset_name + "_test.csv"), "", attribute)

    train_features = np.load(save_dir + "train_features.npy")
    pca2d(train_features, train_dataset.labels, train_dataset.class_names, train_dataset.img_files,
                  ordered_styles,"train", save_dir, dataset_name=dataset_name, n_components=2, fontsize=10)
    pca2d_interactive(train_features, train_dataset.labels, train_dataset.class_names, train_dataset.img_files,
                  ordered_styles,"train", save_dir, dataset_name=dataset_name, n_components=2)
    pca3d(train_features, train_dataset.labels, train_dataset.class_names,
                  ordered_styles,"train", save_dir, dataset_name=dataset_name, n_components=3)


    test_features = np.load(save_dir + "test_features.npy")
    print("filenames", test_dataset.img_files)
    X_red = pca2d(test_features, test_dataset.labels, test_dataset.class_names, test_dataset.img_files,
                  ordered_styles, "test", save_dir, dataset_name=dataset_name, n_components=2, fontsize=10)
    pca2d_interactive(test_features, test_dataset.labels, test_dataset.class_names, test_dataset.img_files,
                      ordered_styles, "test", save_dir, dataset_name=dataset_name, n_components=2)
    pca3d(test_features, test_dataset.labels, test_dataset.class_names, ordered_styles,
          "test", save_dir, dataset_name=dataset_name, n_components=3)

    return X_red


if __name__ == '__main__':
    # For WikiArt
    results_dir = "highest_expl_variance/"
    dataset_name = "WikiArt"
    annotations_dir = "annotations/"
    attribute = "art_style"

    X_red = main(dataset_name, annotations_dir, results_dir, attribute)
