import os
import random

import matplotlib.pyplot as plt
import numpy
from PIL import Image


def class_distribution(plot=True, save_fig=False):
    images_path = os.path.join('..', 'database_limited')
    dirs = os.listdir(images_path)
    dist = {}
    for dir in dirs:
        dist[dir] = len(os.listdir(os.path.join(images_path, dir)))

    if plot:
        plt.figure(figsize=(8, 4))
        bars = plt.bar(dist.keys(), dist.values())
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .01, yval)
        labels = [d for d in dist.keys()]
        plt.xticks(range(len(dist.keys())), labels, rotation=90)
        plt.title("Liczba obrazów przypisanych do poszczególnych klas")
        plt.tight_layout()
        if save_fig:
            plt.savefig('class_distribution.png')
        plt.show()

    return dist


def size_distribution(plot=True, save_fig=False, most_common=0):
    images_path = os.path.join('..', 'database_limited')
    dirs = os.listdir(images_path)
    sizes = {}
    for dir in dirs:
        imgs = os.listdir(os.path.join(images_path, dir))
        for img in imgs:
            im = Image.open(os.path.join(images_path, dir, img))
            if im.size in sizes.keys():
                sizes[im.size] += 1
            else:
                sizes[im.size] = 1

    if most_common > 0:
        most_common_sizes = dict(sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:20])
        to_plot = most_common_sizes
    else:
        to_plot = sizes

    if plot:
        plt.figure(figsize=(8, 4))
        bars = plt.bar(range(len(to_plot.keys())), to_plot.values())
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .01, yval)
        plt.xticks(range(len(to_plot.keys())), to_plot.keys(), rotation=45)
        plt.title("Częstotliwość występowania obrazów o 20 najczęściej występujących rozmiarach")
        plt.tight_layout()
        if save_fig:
            plt.savefig('size_distribution.png')
        plt.show()

    return to_plot


def show_grid(image_list, nrows, ncols, label_list, figsize=(8, 8), hpad=0, save_fig=False, filename=''):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].imshow(image_list[k])
            ax[i, j].set_xlabel(label_list[k])
            ax[i, j].xaxis.set_ticks([])
            ax[i, j].yaxis.set_ticks([])
            k += 1
    fig.tight_layout(h_pad=hpad)
    if save_fig:
        if filename == '':
            plt.savefig('default.png')
        else:
            plt.savefig(filename)
    plt.show()


def show_random_images(rows, cols, save_fig=False):
    images_path = os.path.join('..', 'database_limited')
    dirs = os.listdir(images_path)
    img_list = []
    label_list = []
    for i in range(rows*cols):
        rand_dir = random.choice(dirs)
        rand_img = random.choice(os.listdir(os.path.join(images_path, rand_dir)))
        im = Image.open(os.path.join(images_path, rand_dir, rand_img))
        np_im = numpy.array(im)
        img_list.append(np_im)
        label_list.append(rand_dir)
    show_grid(img_list, rows, cols, label_list=label_list, save_fig=save_fig)
