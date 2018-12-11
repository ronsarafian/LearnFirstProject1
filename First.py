import cv2
import pylab as plt
from ProjectParams import getparams
import numpy as np
import GetData
import DataPreper
import SplitData
import TrainData
import TestData

p = getparams()

np.random.seed(0)     # Seed the random number generator
# p["Data"]["LoadFromCache"] = True
# p["Kmean"]["LoadFromCache"] = True
# p["DataProcess"]["LoadFromCache"] = True


folderList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images, labels = GetData.get_data(p['Data'], folderList)
image_features = DataPreper.data_prepare(p['DataProcess'], images)
train_x_array, train_y_array, test_x_array, test_y_array = SplitData.split_data_train_test(p['Split'], image_features)

linear_svms = []
poly_svms = []
decisions_array = []
decisions_poly_array = []

for i in range(-5, 15):
    p["Train"]["C_Value"] = np.float_power(10, i)
    linear_svm = TrainData.train_data_linear(p['Train'], train_x_array, train_y_array)
    poly_svms = TrainData.train_data_non_linear(p['Train'], train_x_array, train_y_array)
    decisions = TestData.test_linear_svm(p['Test'], linear_svm, test_x_array, test_y_array)
    decisions_poly = TestData.test_poly_svm(p['Test'], poly_svms, test_x_array, test_y_array)
    linear_svms.append(linear_svm)
    decisions_array.append(decisions)
    decisions_poly_array.append(decisions_poly)

