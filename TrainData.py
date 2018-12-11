from sklearn.svm import LinearSVC
from sklearn import datasets, svm
import pickle


def train_data_linear(p, train_x_array, train_y_array):

    if p['LoadFromCache']:
        lin_clf = pickle.load(open(p['CachePath'], "rb"))
    else:
        c = p['C_Value']
        lin_clf = svm.LinearSVC(C=c, max_iter=1000, multi_class='ovr')
        lin_clf.fit(train_x_array, train_y_array)
        pickle.dump(lin_clf, open(p['CachePath'], "wb"))

    return lin_clf


def train_data_non_linear(p, train_x_array, train_y_array):

    non_lin_svms = []
    c = p['C_Value']
    for i in range(p['NumberOfClasses']):
        y_data = []
        for d in range(len(train_y_array)):
            if train_y_array[d] == i:
                y_data.append(1)
            else:
                y_data.append(-1)
        poly_svm = svm.SVC(C=c, kernel='poly', degree=3)
        poly_svm.fit(train_x_array, y_data)
        non_lin_svms.append(poly_svm)

    # pickle.dump(lin_clf, open(p['CachePath'], "wb"))
    return non_lin_svms





# #
# if p['Kmean']['LoadFromCache']:
#     kmean = pickle.load(open(p['Kmean']['CachePath'], "rb"))
# else:
#     x = []
#     # x_total = []
#     for i in range(number_of_labels):
#         number_images_for_test = 10
#         for j in range(number_images_for_test):
#             for k in range(len(image_features[i][j][1])):
#                 if np.random.random() < 1:
#                     x.append(image_features[i][j][1][k])
#     t = np.asarray(x)
#     kmean = KMeans(n_clusters=300, max_iter=50, random_state=0, algorithm="full", precompute_distances=True, n_jobs=-2).fit(t)
#     pickle.dump(kmean, open(p['Kmean']['CachePath'], "wb"))
#
# if p['DataForT']['LoadFromCache']:
#     kmean = pickle.load(open(p['DataForT']['CachePath'], "rb"))
# else:
#     for i in range(number_of_labels):
#         number_images_for_test = 20
#         for j in range(number_images_for_test):
#             x = [0] * 300
#             for k in range(len(image_features[i][j][1])):
#                 sift = image_features[i][j][1][k]
#                 index_of_min_dist = 0
#                 min_dist = scipy.spatial.distance.euclidean(sift, kmean.cluster_centers_[0])
#                 for l in range(1, len(kmean.cluster_centers_)):
#                     dist = scipy.spatial.distance.euclidean(sift, kmean.cluster_centers_[l])#sift * kmeans[i][l]
#                     if dist < min_dist:
#                         min_dist = dist
#                         index_of_min_dist = l
#                 x[index_of_min_dist] += 1
#             x_train.append(x)
#             y_train.append(i)
#     pickle.dump(x_train, open(p['DataForT']['CachePath'], "wb"))