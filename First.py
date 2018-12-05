import skimage.data as skid
import cv2
import pylab as plt
import scipy.misc
import Params
import os
import numpy as Np
import GetData


p = Params

Np.random.seed(0)     # Seed the random number generator
p.Params.LoadFromCache = False
folderList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GetData.GetData(p, folderList)

folderPath = p.Params.BaseDataPath
folders = os.listdir(folderPath)
foldersToWork = folders[0:10]







img = scipy.misc.face()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.imshow(img)
plt.show()

sift = cv2.xfeatures2d.SIFT_create()

step_size = 5
kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                                    for x in range(0, gray.shape[1], step_size)]

img=cv2.drawKeypoints(gray,kp, img)

plt.figure(figsize=(20,10))
plt.imshow(img)
plt.show()

dense_feat = sift.compute(gray, kp)

plt.figure(figsize=(20,10))
plt.imshow(img)
plt.show()
