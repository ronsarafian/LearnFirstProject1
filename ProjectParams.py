import os


def getparams():
    params = {
        "Data": {
            "BaseDataPath": os.path.join("D:\\", "Studies", "Learn", "101_ObjectCategories"),
            "ResizePixelSize": 100,
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedData.pkl"),
            "CacheLablesPath": os.path.join("D:\\", "Studies", "Learn", "CachedDataLables.pkl"),
            "NumberOfImages": 40
        },
        "DataProcess": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedDataSift.pkl"),
            "orientations": 8,
            "pixels_per_cell_x": 16,
            "pixels_per_cell_y": 16,
        },
        "Kmean": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedKmean.pkl"),
        },
        "Split": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedSplitModel.pkl"),
            "C_Value": 1.0,
            "NumberOfImagesForTest": 20,
            "NumberOfImagesForTrain": 20
        },
        "Train": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedTrainModel.pkl"),
            "C_Value": 1.0,
            "NumberOfImages": 20,
            "NumberOfClasses": 10
        },
        "Test": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedTest.pkl"),
            "C_Value": 1.0,
            "NumberOfImages": 20
        }
    }
    return params
