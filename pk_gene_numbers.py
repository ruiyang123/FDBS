import pickle
from data import get_multitask_experiment
import pandas as pd





factors={"Numbers_DIL":4,"Numbers_smoothDIL1":6,"Numbers_smoothDIL2":6,"Numbers_smoothDIL3":6,
         "Numbers_CIL":10,"Numbers_smoothCIL1":6,"Numbers_smoothCIL2":6,"Numbers_smoothCIL3":6}

for factor, task in factors.items():
    file_path = "./{}.csv".format(factor)
    df = pd.read_csv(file_path)
    pkl=get_multitask_experiment(name='numbers_dataset', scenario='domain', tasks=task, verbose=True, exception=True,factor=factor,df=df)
    with open(factor+'.pk','wb') as f:
        pickle.dump(pkl,f)
