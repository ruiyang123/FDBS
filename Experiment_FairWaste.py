import os
import pandas as pd
import pickle
from data import get_multitask_experiment



factors={"CIL_scenario":14,"DIL_scenario":26}

batch_sizes = [8,16,32,64,128]
budgets = [100,200,400,1000]
lrs = [1e-5,5e-5,1e-4]
methods = ["FDBS"]
iters = 300
N = 1
for i in range(N):
    for factor,tasks in factors.items():
        for method in methods:
            for batch_size in batch_sizes:
                for budget in budgets:
                    for lr in lrs:
                        save_path = "{}_{}_{}_{}".format(method,batch_size, budget,lr)
                        if method == "cul":
                            os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --cumulative 1 --batch {} --lr 2.5e-5".format(factor,iters,method,tasks,batch_size))

                        elif method == "naive":
                            os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --lr 4e-5".format(factor,iters,
                                                                                                                                     method,
                                                                                                                                     tasks,batch_size))
                        elif method == "bgd":
                            os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=bgd --tasks {} --batch {}".format(
                                    factor, 300,
                                    method,
                                    tasks, 32))

                        elif method == "er":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --replay=exemplars --budget={} --select-memory random --meta 0 --lr {}".format(
                                    factor,iters, save_path, tasks,batch_size,budget,lr))


                        elif method == "lwf":
                            os.system(
                                "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --reInitOptimizer 1 --batch {} --optimizer=adam --tasks {} --lr 4e-5".format(factor,iters, method,batch_size,
                                                                                                                           tasks))
                        elif method == "erlwf":
                            os.system(
                                "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --optimizer=adam --tasks {} --autolwf --reInitOptimizer 1 --batch {} --lr 4e-5".format(factor, iters, method,
                                                                                                                           tasks,batch_size))

                        elif method == "mlwf":
                            os.system(
                                "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random --lr {}".format(
                                    factor,iters, save_path,batch_size,
                                    tasks,budget,lr))

                        elif method == "oewc":
                            os.system(
                                "python main.py --factor {} --iters {} --ewc --online --batch {} --savepath={} --optimizer=adam --tasks {} --lambda 5000 --lr 4e-5".format(
                                    factor, iters, batch_size,method,
                                    tasks))

                        elif method =="si":
                            os.system(
                                "python main.py --factor {} --iters {} --batch {} --si --savepath={} --optimizer=adam --tasks {} --c 0.1 --lr 4e-5".format(
                                    factor, iters, batch_size, method,
                                    tasks))

                        elif method == "rslwf":
                            os.system(
                                "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random --lr 3e-5 --rs 1".format(
                                    factor,iters, method,batch_size,
                                    tasks,budget))

                        elif method == "rs":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --select-memory random --meta 0 --lr 3e-5".format(
                                    factor,iters, method, tasks,batch_size,budget))

                        elif method == "ps+rs":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --ps --rs 1 --budget={} --select-memory random --meta 0 --lr 3e-5".format(
                                    factor,iters, method, tasks,batch_size,budget))
                        elif method == "FDBS":
                            os.system(
                                "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --remove random --select-memory FDBS --meta 0 --lr {}".format(
                                    factor, iters, save_path, tasks, batch_size, budget,lr))