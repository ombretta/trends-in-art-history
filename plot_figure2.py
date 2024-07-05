# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:15:31 2024

@author: gestaltrevision
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull

import torch
from torch.utils.data import Dataset

# Ordered styles for WikiArt dataset
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
    """
    Custom dataset for loading art images and their labels.
    """

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
        image = torch.load(img_path.replace(".jpg", ".pt"), map_location=self.device).squeeze(0)
        label = self.labels[idx]
        return image.to(torch.float32), label

    def get_class_name(self, idx):
        return self.class_names[idx]


def pca_transform(X, n_components):
    """
    Perform PCA and return the transformed data and explained variance ratios.
    """
    pca_model = PCA(n_components=n_components)
    X_red = pca_model.fit_transform(X)
    explained_variance = [round(e, 2) for e in pca_model.explained_variance_ratio_]
    print("Explained variance ratio", explained_variance)
    return X_red, explained_variance


def get_border_points(X_red):
    """
    Get the border points using Convex Hull for PCA-reduced data.
    """
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
    """
    Get coordinates of painters who have at least `filter_freq` paintings.
    """
    painters_check = []
    painters = []
    for filename in filenames:
        painter = filename.split("/")[1].split("_")[0]
        if painter not in painters_check:
            painters_check.append(painter)
            painters.append(filename)
    print(f"Total painters: {len(painters_check)}")

    coordinates = []
    filtered_painters = []
    for i, painting_filename in enumerate(painters):
        freq = len(filenames[filenames.str.contains(painters_check[i])])
        if freq >= filter_freq:
            filtered_painters.append(painters_check[i])
            xy = from_filename_to_coordinates(painting_filename, X_red, filenames)
            coordinates.append(xy)

    return filtered_painters, coordinates


def add_image(ax, figpath, x, y, zoom=0.15):
    """
    Add an image to a plot at specified coordinates.
    """
    img = mpimg.imread(figpath)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)
    return ax


def pca2d_with_images(X, labels, styles, filenames, ordered_styles, split, savedir, chosen_images,
                      image_dir, n_components=2, fontsize=30, dotsize=5):
    """
    Plot 2D PCA with images on the plot.
    """
    print(f"PCA - {split} set")

    X_red, explained_variance = pca_transform(X, n_components)
    border_points = get_border_points(X_red)

    border_paintings = [from_coordinates_to_filename(point, X_red, filenames) for point in border_points]

    c = [ordered_styles.index(styles[l]) for l in labels]
    fig, ax = plt.subplots(figsize=(80, 30))
    ax.scatter(X_red[:, 0], X_red[:, 1], c=c, cmap='plasma', s=dotsize)

    for image_info in chosen_images:
        figpath, img_x, img_y = image_info.split(" ")
        xy = from_filename_to_coordinates(figpath, X_red, filenames)
        img_x, img_y = float(img_x), float(img_y)
        ax.annotate("", xy=(xy[0], xy[1]), xycoords='data',
                    xytext=(img_x, img_y), textcoords='data',
                    arrowprops=dict(arrowstyle="simple", connectionstyle="angle3,angleA=0,angleB=90",
                                    facecolor='white'), fontsize=fontsize - 20)
        add_image(ax, os.path.join(image_dir, figpath.split("/")[1]), img_x, img_y, zoom=0.15)

    ax.set_xlabel("PC 1", fontsize=fontsize + 30)
    ax.set_ylabel("PC 2", fontsize=fontsize + 30)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_ylim(bottom=-30, top=40)
    ax.set_xlim(left=-40, right=40)

    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(axis='both', which='both', length=0, labelleft=True, labelbottom=True)

    ax = add_legend(ax, ordered_styles, labels, styles, fontsize=fontsize)

    plt.savefig(os.path.join(savedir, f"{split}_PCs_with_images.png"), bbox_inches='tight')
    np.savetxt(os.path.join(savedir, f"{split}_explained_variance1_2.csv"), explained_variance, delimiter=",")

    return X_red


def add_legend(ax, ordered_styles, labels, label_names, fontsize=18):
    """
    Add a custom legend to the plot.
    """
    ordered_label_names = sorted(label_names, key=lambda x: ordered_styles.index(x))
    unique_labels = np.unique(labels)
    legend_labels = [ordered_label_names[i] for i in unique_labels]
    colors = [plt.cm.plasma(i / len(label_names)) for i in unique_labels]

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=30) for color in colors]

    ax.legend(handles, [l.replace("_", " ") for l in legend_labels], loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=fontsize, title_fontsize=fontsize)

    plt.subplots_adjust(right=0.75)
    return ax


def show_image(image_path):
    """
    Display an image from a file path.
    """
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def main(dataset_name, annotations_dir, image_dir, save_dir, attribute):
    """
    Main function to run the PCA and plotting for a given dataset.
    """
    plt.rcParams['font.family'] = 'Times New Roman'

    ordered_styles = styles_order[dataset_name]
    test_dataset = ArtDataset(os.path.join(annotations_dir, f"{dataset_name}_test.csv"), "", attribute)

    with open(image_dir+"/list_to_plot.txt", "r") as f:
        images_to_plot = [line.strip() for line in f]

    test_features = np.load(os.path.join(save_dir, "test_features.npy"))

    pca2d_with_images(test_features, test_dataset.labels, test_dataset.class_names, test_dataset.img_files,
                      ordered_styles, "test", save_dir, images_to_plot, image_dir,
                      n_components=2, fontsize=50, dotsize=30)


if __name__ == '__main__':
    results_dir = "highest_expl_variance/"
    dataset_name = "WikiArt"
    annotations_dir = "annotations/"
    attribute = "art_style"
    image_dir = "images_to_plot/"

    main(dataset_name, annotations_dir, image_dir, results_dir, attribute)
