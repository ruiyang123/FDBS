import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

dataset = "fairwaste"


scenarios = {"CLass-Inc": ["CIL_scenario_result"],"Domain-CLass-Inc":["DIL_scenario_result"]
}

methods = ["Fine Tuning","ER","GSS","MIR","OCS","CBRS","i.i.d. offline","New"]

results = {}
for sce in scenarios.keys():
    results[sce] = {}
    for method in methods:
        results[sce][method] = []


for sce, result_folders in scenarios.items():
    for result_folder in result_folders:
        base_folder = os.path.join("./", result_folder)
        for method in methods:
            method_folder = os.path.join(base_folder, method)
            experiments = os.listdir(method_folder)
            for experiment in experiments:
                result_files = os.listdir(os.path.join(method_folder, experiment))
                result_csv = [file for file in result_files if ".csv" in file]
                result = pd.read_csv(os.path.join(method_folder, experiment, result_csv[0]), header=None)
                if sce =="CLass-Inc" and method!="i.i.d. offline":
                    result = result.values[0:-1,0:-1]
                else:
                    result = result.values
                task_num = len(result)
                av_accuracy = np.mean(result, axis=1)
                results[sce][method].append(av_accuracy)
                print(sce,result_folder,method,experiment,av_accuracy)

np.save("./results_fw_cl.npy",results)


results = np.load('results_fw_cl.npy',allow_pickle='TRUE').item()
df = pd.DataFrame()


for sce in scenarios.keys():
    for method in methods:
        all_ex_res = np.array(results[sce][method])
        av_ = np.mean(all_ex_res,0)
        std_ = np.std(all_ex_res,0)
        results[sce][method] = av_

        for task in range(len(av_)):
            if method=="New":
                method="OURS"
            # if method == "bgd":
            #     method = "FOO-VB"
            row = {"method":method,"task_num":int(task),"Mean-accuracy":av_[task],"scenario":sce}
            df = df.append(row,ignore_index=True)

df.to_csv('./result.csv')

paper_rc = {'lines.linewidth': 4, 'lines.markersize': 10}
sns.set_context("paper", rc = paper_rc)
fig, axs = plt.subplots(1,2,figsize=(18,6))


sns.lineplot(ax=axs[0],x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="CLass-Inc"])
#axs[2].get_legend().remove()
axs[0].set_title("Class-Inc",fontsize=20)
axs[0].set(xlabel=None)
axs[0].set(ylabel=None)
axs[0].set(xlabel=None)
axs[0].xaxis.set_ticks(np.arange(0,14,1))
axs[0].set_ylabel("Average-Accuracy",fontsize=20)

sns.lineplot(ax=axs[1],x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="Domain-CLass-Inc"])

axs[1].get_legend().remove()
axs[1].set_title("Domain-Class-Inc",fontsize=20)
axs[1].set(xlabel=None)
axs[1].set(ylabel=None)
axs[1].set(xlabel=None)
axs[1].xaxis.set_ticks(np.arange(0,26,1))


fig.text(0.5, 0.01, 'Task', ha='center',fontsize=18)


fig.savefig("./FDBS_result_memory_method2.svg")
