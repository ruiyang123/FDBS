import os
import pandas as pd

factors={"PACS_DIL":4}

batch_size = 32
iters = 4600

for factor, tasks in factors.items():
    methods = ["aocl+fo", "aocl+lwf"]#["aocl","naive","lwf","aocl+co","aocl+fo", "aocl+lwf"]
    for method in methods:
        if method == "aocl+fo":
            os.system("python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --aocl --using-feature-output".format(factor,iters,batch_size,tasks))

        if method == "aocl+co":
            os.system("python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --aocl --using-classification-output".format(factor,iters,batch_size,tasks))

        if method == "aocl":
            os.system("python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --aocl --using-feature-output --using-classification-output".format(factor,iters,batch_size,tasks))

        if method == "aocl+lwf":
            os.system(
                "python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --lwf --aocl --using-feature-output".format(
                    factor, iters, batch_size, tasks))
        if method == "aocl+autolwf":
            os.system(
                "python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --autolwf --aocl --using-feature-output".format(
                    factor, iters, batch_size, tasks))
        if method == "naive":
            os.system(
                "python AOCL.py --factor {} --iters {} --batch-size {} --task-num {}".format(
                    factor, iters, batch_size, tasks))
        if method == "lwf":
            os.system(
                "python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --lwf".format(
                    factor, iters, batch_size, tasks))
        if method == "autolwf":
            os.system(
                "python AOCL.py --factor {} --iters {} --batch-size {} --task-num {} --autolwf".format(
                    factor, iters, batch_size, tasks))