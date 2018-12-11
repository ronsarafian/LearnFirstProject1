import os
import cv2
import pickle


def get_data(p, folder_list):
    if p['LoadFromCache']:
        images = pickle.load(open(p['CachePath'], "rb"))
        labels = pickle.load(open(p['CacheLablesPath'], "rb"))
    else:
        folder_path = p['BaseDataPath']
        images, labels = open_folders(p, folder_path, folder_list)
        pickle.dump(images, open(p['CachePath'], "wb"))
        pickle.dump(labels, open(p['CacheLablesPath'], "wb"))
    return images, labels


def open_folders(p, folder_path, folder_list):
    folders = os.listdir(folder_path)
    labels = []
    images = []
    for fo in folder_list:
        folder_path_of_image = os.path.join(folder_path, folders[fo])
        image, label = open_files(p, folder_path_of_image, folders[fo])
        images.append(image)
        labels.append(label)
    return images, labels


def open_files(p, folder_path_of_images, label):
    images = []
    labels = []
    files = os.listdir(folder_path_of_images)
    number_of_files = len(files)
    if number_of_files > p['NumberOfImages']:
        number_of_files = p['NumberOfImages']
    for fi in range(number_of_files):
        labels.append(label)
        image_path = os.path.join(folder_path_of_images, files[fi])
        dest_path = os.path.join(folder_path_of_images, "Scaled_" + files[fi])
        image = resize_image(p, image_path, dest_path)
        images.append(image)
    return images, labels


def resize_image(p, image_path, DestPath):
    src_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    scaled_image = cv2.resize(gray_image, (p['ResizePixelSize'], p['ResizePixelSize']), interpolation=cv2.INTER_LANCZOS4)
    return scaled_image
    # images.append(scaledImage)
