from skimage.feature import hog
import pickle


def data_prepare(p, images):
    if p['LoadFromCache']:
        image_feats = pickle.load(open(p['CachePath'], "rb"))
    else:
        image_feats = process_all_labels(p, images)
        pickle.dump(image_feats, open(p['CachePath'], "wb"))
    return image_feats


def process_all_labels(p, images):
    image_feats = []
    for image in images:
        image_feat = process_label(p, image)
        image_feats.append(image_feat)
    return image_feats


def process_label(p, images):
    image_feats = []
    for image in images:
        image_feat = get_data_hog(p, image)
        image_feats.append(image_feat)
    return image_feats


def get_data_hog(p, image):
    image_feat = hog(image, orientations=p['orientations'],
                     pixels_per_cell=(p['pixels_per_cell_x'], p['pixels_per_cell_x']),
                     cells_per_block=(1, 1), feature_vector=True)
    return image_feat

#
#
# def get_dense_sift(image):
#     gray = image
#     sift = cv2.xfeatures2d.SIFT_create()
#     step_size = 5
#     kp = []
#     kp1 = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
#           for x in range(0, gray.shape[1], step_size)]
#     kp2 = [cv2.KeyPoint(x, y, 10) for y in range(0, gray.shape[0], 10)
#            for x in range(0, gray.shape[1], 10)]
#     kp3 = [cv2.KeyPoint(x, y, 20) for y in range(0, gray.shape[0], 20)
#            for x in range(0, gray.shape[1], 20)]
#     # kp4 = [cv2.KeyPoint(x, y, 3) for y in range(0, gray.shape[0], 3)
#     #        for x in range(0, gray.shape[1], 3)]
#
#     for i in range(len(kp1)):
#         kp.append(kp1[i])
#     for i in range(len(kp2)):
#         kp.append(kp2[i])
#     for i in range(len(kp3)):
#         kp.append(kp3[i])
#     dense_feat = sift.compute(gray, kp)
#     return dense_feat
