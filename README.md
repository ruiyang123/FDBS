# ERLWF

## Requirements
```bash
$ pip3 install -r requirements.txt
```

## Data Preparation

PACS: you can get the PACS dataset from [https://github.com/MachineLearning2020/Homework3-PACS](https://github.com/MachineLearning2020/Homework3-PACS)

Cifar100: In our experiments we use the .png version of cifar100.
You can download the dataset by using the following commands :

```bash
$ pip install cifar2png
$ cifar2png cifar100superclass path/to/cifar100png
```
Before you start the experiments, it is neccesaire to place the two datasets under the directory "./datasets/"

Generally the directories for PACS looks like: 

```bash
./datasets/PACS
├── art_painting
│   ├── dog
...
...
```

```bash
For cifar100 : 

./datasets/cifar100
├── train
│   ├── fish
│   │   ├── flatfish
...
├── test
...
```

```bash
For FairWaste DataSet : 
```

Download from the following sites our prepared scenario files [Class-Incremental scenario](https://drive.google.com/file/d/1SuX8E_6TLlgQ1txjk6x-VvWmwj9zFUFL/view?usp=sharing) and 
[Domain-Incremental scenario](https://drive.google.com/file/d/1SuX8E_6TLlgQ1txjk6x-VvWmwj9zFUFL/view?usp=sharing)

Or 

You can also prepare your scenario by creating a .csv file such as [./CIL_scenario.csv](./CIL_scenario.csv). Then run 
```bash
python pk_gene_FW.py
```
To create the scenario to be used for training.

## Run Methods
```bash
For Experience Replay :
python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --rs 1 --budget={} --select-memory random --meta 0 --lr 5e-5"
```

```bash
For FDBS :
python main.py --factor scenario_name --iters 300 --savepath savepath --tasks task_num --repaly=exemplars --budget budget --select-memory FDBS
```



## Run Experiment

```bash
For FairWaste DataSet :
python Experiment_FairWaste.py
```





