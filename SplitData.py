import numpy as np


def split_data_train_test(p, image_features):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    number_of_labels = len(image_features)

    for i in range(number_of_labels):
        number_images_for_train = p['NumberOfImagesForTest']
        for j in range(number_images_for_train):
            y_train.append(i)
            x_train.append(image_features[i][j])

    for i in range(number_of_labels):
        number_images_for_test = len(image_features[i])
        for j in range(p['NumberOfImagesForTest'], number_images_for_test):
            y_test.append(i)
            x_test.append(image_features[i][j])

    train_x_array = np.asarray(x_train)
    train_y_array = np.asarray(y_train)
    test_x_array = np.asarray(x_test)
    test_y_array = np.asarray(y_test)
    return train_x_array, train_y_array, test_x_array, test_y_array
