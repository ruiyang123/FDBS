import os
import pandas as pd
import pickle
from data import get_multitask_experiment



factors={"CIL_scenario":22}#{"CIL_scenario":22,"CL_scenario":9}

batch_size = 16
budgets = [100,400,700]
iters = 3000
N = 3

for budget in budgets:
    for i in range(N):
        for factor,tasks in factors.items():
            methods = ["FDBS","FDBS+lowei","FDBS+highei"]
            for method in methods:
                save_path = "{}_{}".format(method, budget)
                if method == "cbrs-distill":
                    os.system(
                        "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --remove CBRS --select-memory random --lr 3e-5 --rs 1".format(
                            factor, iters, save_path, batch_size,
                            tasks, budget))
                if method == "cbrs":
                    os.system(
                        "python main.py --factor {} --iters {} --replay=exemplars  --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --remove CBRS --select-memory random --lr 3e-5 --rs 1".format(
                            factor, iters, save_path, batch_size,
                            tasks, budget))
                elif method == "rslwf":
                    os.system(
                        "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random --lr 3e-5 --rs 1".format(
                            factor,iters, save_path,batch_size,
                            tasks,budget))

                elif method == "rs":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --select-memory random --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))

                elif method == "rs+lowei":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove lowEI --select-memory random --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))
                elif method == "rs+highei":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove highEI --select-memory random --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))
                elif method == "rs+lowei+distill":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --distill --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove lowEI --select-memory random --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))
                elif method == "rs+highei+distill":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --distill --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove highEI --select-memory random --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))
                elif method == "FDBS":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove random --select-memory FDBS --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))

                elif method == "FDBS+highei":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove highEI --select-memory FDBS --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))
                elif method == "FDBS+lowei":
                    os.system(
                        "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove lowEI --select-memory FDBS --meta 0 --lr 3e-5".format(
                            factor,iters, save_path, tasks,batch_size,budget))