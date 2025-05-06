# JIT-DCK Replication Package

This repository contains the implementation code and the methodological guidelines to replicate the study described in the paper "Just-In-Time Defect Prediction and Localization via Kolmogorov–Arnold Networks".


## 🎯 Overview
We propose a unified approach for real-time defect identification (JIT-DP) and precise line-level localization (JIT-DL) in agile development, addressing unstructured code changes and complex feature interactions.To better illustrate this approach, [an overview diagram](Overview.jpg) is provided.Experiments are conducted on [JIT-Defect4J](https://github.com/jacknichao/JIT-Fine) - a multi-granularity dataset with 21 Java projects containing commit-level labels, line-level annotations, and expert-designed features.


## 🔧 Environment Setup

Run the following command under your python environment

```shell
pip install requirements.txt
```


## 🔍 Experiment Result Replication Steps

Note: To mitigate potential threats to internal validity, we utilize the source code from the original studies rather than independently implementing the benchmark methods. By adopting the default optimized parameters and operational protocols outlined in the respective papers, we ensure the accuracy of our replication efforts. Additionally, specific evaluation results of the analyzed baselines are referenced from the work of Chen et al. (https://github.com/JIT-A/JIT-Smart).

### 📈 RQ1: How effective is our JIT-DCK model in JIT defect prediction? Does it offer higher cost-effectiveness compared to state-of-the-art defect prediction methods?

Step 1: Run the following two (*.ipynb) files to convert the line-level data annotation format.
  
  ```shell
./JITDCK/process and label defect code lines/extract defect lines.ipynb

./JITDCK/process and label defect code lines/label defect lines.ipynb
  ```
Step 2: Train and evaluate our JIT-DCK and JIT-Smart in the JIT-DP task.
  ```shell
sh train_jitdck.sh
sh train_jitsmart.sh
  ```


### 📊 RQ2: How effective is our JIT-DCK model in JIT defect localization? Does it demonstrate greater cost-effectiveness when compared with the most advanced defect localization methods?


Step 1: Train and evaluate our JIT-DCK and JIT-Smart in the JIT-DL task.
  ```shell
sh train_jitdck.sh
sh train_jitsmart.sh
  ```


### 🌐 RQ3: How does the accuracy of JIT-DCK compare to the SOTA models in cross-project experiments?

Step 1: Generate the cross-project data for our JIT-DCK and JIT-Smart.
  ```shell
python jitdck cross prj data generate.py
python jitsmart cross prj data generate.py
  ```

Step 2: Train and evaluate our JIT-DCK and JIT-Smart in the JIT-DP and JIT-DL tasks under cross-project settings.
  ```shell
sh train_jitdck_cross_prj.sh
sh train_jitsmart_cross_prj.sh
  ```

### 📚 RQ4: What impact do the parameter settings of code defect prediction component in JIT-DCK have on the model's performance?

You need to manually change the parameter configuration and run the JIT-DP and JIT-DL tasks.

  ```shell
sh train_jitdck.sh
  ```

