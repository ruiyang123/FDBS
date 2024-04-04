import pickle
from data import get_multitask_experiment
import pandas as pd





factors={"imagenet_CIL":10,"imagenet_smoothCIL1":10}

for factor, task in factors.items():
    file_path = "./{}.csv".format(factor)
    df = pd.read_csv(file_path)
    pkl=get_multitask_experiment(name='imagenet_dataset', scenario='domain', tasks=task, verbose=True, exception=True,factor=factor,df=df)
    with open(factor+'.pk','wb') as f:
        pickle.dump(pkl,f)
