import os


# class Params:

def getparams():

    params = {
        "Data": {
            "BaseDataPath": os.path.join("D:\\", "Studies", "Learn", "101_ObjectCategories"),
            "ResizePixelSize": 100,
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedData.pkl")
        }
    }
    return params





