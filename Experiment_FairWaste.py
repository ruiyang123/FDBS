import os
import pandas as pd
import pickle
from data import get_multitask_experiment



factors={"CIL_scenario":14,"DIL_scenario":26}

batch_sizes = [8,16,32,64]
budgets = [400,800]
lrs = [1e-5]
methods = ["FDBS+IWL","FDBS","ER","MIR","cul","naive","oewc","si"]
iters = 2000
N = 3
for i in range(N):
    for factor,tasks in factors.items():
        for method in methods:
            for batch_size in batch_sizes:
                for budget in budgets:
                    for lr in lrs:
                        save_path = "{}_{}_{}_{}".format(method,batch_size, budget,lr)
                        if method == "cul":
                            os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --cumulative 1 --batch {} --lr {}".format(factor,iters,save_path,tasks,batch_size,lr))

                        elif method == "naive":
                            os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --lr {}".format(factor,iters,
                                                                                                                                     save_path,
                                                                                                                                     tasks,batch_size,lr))
                        elif method == "oewc":
                            os.system(
                                "python main.py --factor {} --iters {} --ewc --online --batch {} --savepath={} --optimizer=adam --tasks {} --lambda 5000 --lr {}".format(
                                    factor, iters, batch_size,save_path,
                                    tasks,lr))

                        elif method =="si":
                            os.system(
                                "python main.py --factor {} --iters {} --batch {} --si --savepath={} --optimizer=adam --tasks {} --c 0.1 --lr 4e-5".format(
                                    factor, iters, batch_size, save_path,
                                    tasks))

                        elif method == "ER":
                            os.system(
                                "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random --lr {} --rs 1".format(
                                    factor,iters, save_path,batch_size,
                                    tasks,budget,lr))

                        elif method == "MIR":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={}  --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --select-memory random --meta 0 --lr 3e-5".format(
                                    factor,iters, save_path, tasks,batch_size,budget))

                        elif method == "ps+rs":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --ps --rs 1 --budget={} --select-memory random --meta 0 --lr 3e-5".format(
                                    factor,iters, save_path, tasks,batch_size,budget))
                        elif method == "FDBS":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove random --select-memory FDBS --meta 0 --lr {}".format(
                                    factor, iters, save_path, tasks, batch_size, budget,lr))
                        elif method == "FDBS+IWL":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove random --select-memory FDBS --meta 0 --lr {} --iwl 1".format(
                                    factor, iters, save_path, tasks, batch_size, budget,lr))