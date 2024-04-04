import os
import pandas as pd
import pickle
from data import get_multitask_experiment



factors={"Numbers_DIL":4,"Numbers_smoothDIL1":6,"Numbers_smoothDIL2":6,
          "Numbers_CIL":10,"Numbers_smoothCIL1":6,"Numbers_smoothCIL2":6}

#factors={"Numbers_DIL":4}


batch_size = 16
budget = 1000
iters = 10000
N = 1
for i in range(N):
    for factor,tasks in factors.items():
        methods =  ["bgd"] #,"naive", "erlwf", "cul", "oewc", "si", "er", "lwf", "mlwf"]
        for method in methods:
            if method == "cul":
                os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --cumulative 1 --batch {}".format(factor,iters,method,tasks,batch_size))


            elif method == "naive":
                os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1".format(factor,iters,
                                                                                                                         method,
                                                                                                                         tasks,batch_size))
            elif method == "bgd":
                os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=bgd --tasks {} --batch {}".format(
                        factor, 3000,
                        method,
                        tasks, 32))

            elif method == "er":
                os.system(
                    "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --replay=exemplars --budget={} --select-memory random --meta 0".format(
                        factor,iters, method, tasks,batch_size,budget))


            elif method == "lwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --reInitOptimizer 1 --batch {} --optimizer=adam --tasks {} ".format(factor,iters, method,batch_size,
                                                                                                               tasks))
            elif method == "erlwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --optimizer=adam --tasks {} --autolwf --batch {}".format(factor, iters, method,
                                                                                                               tasks,batch_size))

            elif method == "mlwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random".format(
                        factor,iters, method,batch_size,
                        tasks,budget))

            elif method == "oewc":
                os.system(
                    "python main.py --factor {} --iters {} --ewc --online --batch {} --savepath={} --optimizer=adam --tasks {} --lambda 5000".format(
                        factor, iters, batch_size,method,
                        tasks))

            elif method =="si":
                os.system(
                    "python main.py --factor {} --iters {} --batch {} --si --savepath={} --optimizer=adam --tasks {} --c 0.95".format(
                        factor, iters, batch_size, method,
                        tasks))
