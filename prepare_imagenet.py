import os
import numpy as np
import pandas as pd
import shutil
import glob
import random

def get_distribution(task_num,categories,mode):
    distribution = {}
    if mode=="uniform":
        for cat in categories:
            distribution[cat] = [1./task_num]*task_num
    elif mode == "class_inc":
        dist = np.ones((len(categories),task_num))*(0.1/(task_num-1))
        for i,cat in enumerate(categories):
            dist[i, int(i/10)] = 0.9
            distribution[cat] = list(dist[i,:])
    elif mode == "random":
        for cat in categories:
            dist = [np.random.random() for i in range(task_num)]
            k = sum(dist)
            dist = [i/k for i in dist]
            distribution[cat] = dist
    return distribution




def get_dataset_info(source):
    df = pd.DataFrame()
    categories = os.listdir(source)
    categories_select = random.choices(categories,k=100)
    

    
    for cat in categories_select:
        imgs = os.listdir(os.path.join(source,cat))
        imgs_select = random.choices(imgs,k=500)
        for img in imgs_select:
            if np.random.rand() > 0.8:
                row = {"path": os.path.join(source,cat,img),"class":cat, "type":"test"}
            else:
                row = {"path": os.path.join(source,cat, img),  "class": cat,
                       "type": "train"}
            df = df.append(row,ignore_index=True)

    df.to_csv("./imagenet_info.csv",index=False)





def class_split(df,task_num,mode):
    classes = list(df["class"].unique())
    dist_train = get_distribution(task_num,classes,mode)
    dist_test = get_distribution(task_num,classes,mode="uniform")

    dists = {}
    dists["train"] = dist_train
    dists["test"] = dist_test

    def class_dist_split(series, dists):
        class_ = series["class"]
        type_ = series["type"]
        dist_ = dists[type_][class_]
        a = np.random.random()

        start = 0
        stop = 0
        for i, d in enumerate(dist_):
            stop = start + d
            if a > start and a < stop:
                return i + 1
            start = stop
    df["task"] = df.apply(class_dist_split,args = (dists,),axis=1)
    return df








if __name__ == "__main__":
    source = r"D:\donwloads\Imagenet"
    #get_dataset_info(source)
    info_path = "./imagenet_info.csv"
    imagenet_info = pd.read_csv(info_path)

    # CIL
    df = class_split(df=imagenet_info,task_num=10,mode="class_inc")
    df.to_csv("./imagenet_CIL.csv")

    # smooth-CIL
    df = class_split(df=imagenet_info,task_num=10,mode="random")
    df.to_csv("./imagenet_smoothCIL1.csv")








