import multiprocessing
import os
import numpy as np
import tensorflow as tf
import cv2


def index_images(directory):
    dir_name = os.path.basename(directory)
    filenames = []
    labels = []
    walk = os.walk(directory)
    for _, _, files in walk:
        for file in files:
            img = cv2.imread(f"{directory}/{file}")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            filenames.append(img)
            labels.append(int(dir_name))
    return filenames, labels


def classify_img_dir(directory):
    sub_dirs = []
    for subdir in tf.io.gfile.listdir(directory):
        if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
            if subdir.endswith("/"):
                subdir = subdir[:-1]
            sub_dirs.append(subdir)
    if len(sub_dirs) == 0:
        print("!!! Empty resource directory")
        exit()

    pool = multiprocessing.pool.ThreadPool()
    results = []
    for dir_path in (tf.io.gfile.join(directory, subdir) for subdir in sub_dirs):
        results.append(
            pool.apply_async(
                index_images,
                (dir_path,),
            )
        )
    filepaths = []
    labels_list = []
    for res in results:
        partial_filepaths, partial_labels = res.get()
        labels_list += partial_labels
        filepaths += partial_filepaths
    pool.close()
    pool.join()
    filepaths = np.array(filepaths)
    labels_list = np.array(labels_list)
    return filepaths, labels_list


def get_vn_chars_xy(resource_path):
    if not os.path.exists(resource_path):
        print("Dataset resource not found")
        exit()

    print("... Downloading vn chars")
    dataset_x, dataset_y = classify_img_dir(resource_path)
    return dataset_x, dataset_y
