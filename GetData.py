import os
import cv2
import pickle
import Params

images = []
lables = []

def GetData(p, FolderList):
    if (p.Params.LoadFromCache):
        images = pickle.load(open(p.Params.CachePath, "rb"))
        folderPath = p.Params.BaseDataPath
    else:
        openFolders(p, FolderList)

def openFolders(p, FolderList):
    folderPath = p.Params.BaseDataPath
    folders = os.listdir(folderPath)
    for fo in FolderList:
        lables.append(folders[fo])
        folderPathOfImage = os.path.join(folderPath, folders[fo])
        OpenFiles(p,folderPathOfImage)
    pickle.dump(images, open(p.Params.CachePath, "wb"))



def OpenFiles(p, FolderPathOfImages):
    files = os.listdir(FolderPathOfImages)
    for fi in range(20):
        imagePath = os.path.join(FolderPathOfImages, files[fi])
        destPath = os.path.join(FolderPathOfImages, "Scaled_" +files[fi])
        ResizeImage(p, imagePath, destPath)

def ResizeImage(p, ImagePath, DestPath):
    srcImage = cv2.imread(ImagePath)
    grayImage = cv2.cvtColor(srcImage, cv2.COLOR_RGB2GRAY)
    scaledImage = cv2.resize(grayImage, (p.Params.ResizePixelSize, p.Params.ResizePixelSize), interpolation = cv2.INTER_LANCZOS4)
    images.append(scaledImage)
