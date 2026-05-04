# robust-decision-systems-under-messy-tabular-data
Portfolio ml project on robust models on tabular data


## Dataset
Home Credit Default Risk[1](#references) from [link](https://www.kaggle.com/c/home-credit-default-risk)


### Ad Hoc Usage
The original dataset was published in relation to a prediction on AUROC challenge on Kaggle website. \
It comprised of an unlabeled test set file "application_test.csv", meant to contain test overfitting for fairness purposes. \
In this project such file is not used and test data come from a split on training data in the "application_train" file. \
Also, here models get evaluated on prediction Accuracy.


## Versioning
- Python==3.12.0 ([doc](https://docs.python.org/3.12/))
- matplotlib==3.10.8 ([doc](https://matplotlib.org/stable/index.html))
- numpy==2.4.2 ([doc](https://numpy.org/doc/2.4/reference/index.html#reference))
- pandas==2.3.3 ([doc](https://pandas.pydata.org/pandas-docs/version/2.3/index.html))
- scikit-learn==1.8.0 ([doc](https://scikit-learn.org/stable/api/index.html))
- torch==2.5.1+cu121 ([doc](https://docs.pytorch.org/docs/2.5/))([pkg source](https://download.pytorch.org/whl/cu121)( by adding the "--extra-index-url" option))
- apscheduler==3.11.2 ([doc](https://apscheduler.readthedocs.io/en/3.x/userguide.html))
- pytest==9.0.2 ([doc](https://app.readthedocs.org/projects/pytest/builds/?version__slug=9.0.x))

All packages except Python can be installed with: 
```
pip install -r "requirements.txt"
```


## ELT
Data coming from the 7 raw files
- application_train.csv
- bureau_balance.csv
- bureau.csv
- credit_card_balance.csv
- installments_payments.csv
- POS_CASH_balance.csv
- previous_application.csv

get transformed and saved into the "dataset.csv" file. \
A state variable signals the need to perform such procedure, at the beginning and every time an update loads new data on the raw files. 

### Transform
Features of secondary tables get collapsed and aggregated, following the tree structure of the database, and mean, standard deviation or count get merged to parents:
```
- bureau_balance: 
    1. "MONTHS_BALANCE" -> "BB_MB_MEAN", "BB_MB_STD", "BB_COUNT" (count)

- bureau:
    1. "DAYS_ENDDATE_FACT" -> "B_ENDDATE_MEAN", "B_ENDDATE_STD"
    2. "CNT_CREDIT_PROLONG" -> "B_PROLONG_MEAN", "B_PROLONG_STD"
    3. "AMT_CREDIT_SUM" -> "B_CREDIT_MEAN", "B_CREDIT_STD"
    4. "AMT_CREDIT_SUM_DEBT" -> "B_DEBT_MEAN", "B_DEBT_STD"
    5. "BB_MB_MEAN" -> "BB_MB_MEAN"
    6. "BB_MB_STD" -> "BB_MB_STD" (mean)
    7. "BB_COUNT" -> "BB_COUNT" (mean), "B_COUNT" (count non 0)

- POS_CASH_balance:
    1. "SK_DPD" -> "PCB_DPD_MEAN"
    2. "SK_DPD_DEF" -> "PCB_DPD_DEF_MEAN"

- installments_payments:
    1. "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT" -> "DAYS_INSTALLMENT_DELAY" (diff) -> "IP_DELAY_MEAN", "IP_DELAY_STD"
    2. "AMT_PAYMENT", "AMT_INSTALMENT" -> "AMT_PAYMENT_DIFF" (diff) -> "IP_AMT_DIFF_MEAN", "IP_AMT_DIFF_STD"

- credit_card_balance:
    1. "AMT_BALANCE" -> "CCB_AMT_BALANCE_MEAN", "CCB_AMT_BALANCE_STD"
    2. "SK_DPD" -> "CCB_SK_DPD_MEAN", "CCB_SK_DPD_STD"
    3. "SK_DPD_DEF" -> "CCB_SK_DPD_DEF_MEAN", "CCB_SK_DPD_DEF_STD"

- previous_application:
    1. "AMT_ANNUITY" -> "PA_AMT_ANNUITY_MEAN", "PA_AMT_ANNUITY_STD"
    2. "AMT_APPLICATION" -> "PA_AMT_APPLICATION_MEAN", "PA_AMT_APPLICATION_STD"
    3. "AMT_CREDIT" -> "PA_AMT_CREDIT_MEAN", "PA_AMT_CREDIT_STD"
    4. "AMT_DOWN_PAYMENT" -> "PA_AMT_DOWN_PAYMENT_MEAN", "PA_AMT_DOWN_PAYMENT_STD"
    5. "NAME_YIELD" -> "PA_NAME_YIELD_MEAN"
    6. "PCB_DPD_MEAN" -> "PCB_DPD_MEAN"
    7. "PCB_DPD_DEF_MEAN" -> "PCB_DPD_DEF_MEAN"
    8. "IP_DELAY_MEAN" -> "IP_DELAY_MEAN"
    9. "IP_DELAY_STD" -> "IP_DELAY_STD" (mean)
    10. "IP_AMT_DIFF_MEAN" -> "IP_AMT_DIFF_MEAN"
    11. "IP_AMT_DIFF_STD" -> "IP_AMT_DIFF_STD" (mean)
    12. "CCB_AMT_BALANCE_MEAN" -> "CCB_AMT_BALANCE_MEAN"
    13. "CCB_AMT_BALANCE_STD" -> "CCB_AMT_BALANCE_STD" (mean)
    14. "CCB_SK_DPD_MEAN" -> "CCB_SK_DPD_MEAN"
    15. "CCB_SK_DPD_STD" -> "CCB_SK_DPD_STD" (mean)
    16. "CCB_SK_DPD_DEF_MEAN" -> "CCB_SK_DPD_DEF_MEAN"
    17. "CCB_SK_DPD_DEF_STD" -> "CCB_SK_DPD_DEF_STD" (mean)
```
"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY" are the only selected features of "application_train.csv", with "SK_ID_CURR" that acts as row id and "TARGET" which is indeed the target of prediction, both to be removed to prevent leakage and present only to track data. \
"dataset.csv" features follow a schema saved in the FEATURE_SCHEMA variable.


## Preprocessing
Target class y is named "TARGET". \
y has 0 (0%) null values: no row has been filtered in "application_train.csv". \
This step is separated from ELT transformation because it relies on a specific instance of training data, to be consistent with the current model at use, rather than on raw data updates. \
During training, a new preprocessor for the current training data gets created. \
Once a model gets deployed, such preprocessor gets saved too. \
At inference, the same preprocessor gets loaded. 

### Preprocessor Algorithm
The defined preprocess procedure, receives the training set, without SK_ID_CURR and TARGET columns, and returns both the trained preprocessor and the processed training data. \
To handle NaN or Null values, FEATURE_SCHEMA divides features into "mean", "zero" and "binary", to specify whether to perform imputation on the column mean, padding to 0 or to add an "UNKNOWN" label for values not in {0, 1}. \
Both "mean" and "zero" features get then scaled, while "binary" 3 ({0,1,UNKNOWN}) features get one-hot encoded.


## Models
Models are listed, with thier training information, in a MODEL_REGISTRY variable. \
Specifically, they have a "class", an "arch_params" and a "training_params" sections, specifying model Hp Class prototype, architectural parameters defining the specific Hp Class and training parameters as learning rate, batch size and epochs. 

### PriorModel
Defined as a Pytorch Module subclass, when coupled with argmax function always return the Mode for the training TARGET values. \
It does need no input feature at inference time. \
Gets used as a baseline to compare other models metrics.

### LinearModel
Defined as a Pytorch Module subclass, when coupled with a softmax function is equivalent to a Logistic Regression for all intents and purposes.

### MLP
Defined as a Pytorch Module subclass, is a 2 Layer Perceptron. \
The intermediate dimension is fixed to 64.


## Updating
Updates are stored in a single json, so the assumption is that the load function would append new incoming info to the same json. \
When the update pipeline actually updates raw tables, ELT transformation gets marked as obsolete. \
To prevent data leakage, once a row receives its TARGET, it gets sealed and no further update on it will be applied. \
To provide an up to date inference, if, after the update, a row is still missing the TARGET, its prediction gets removed so it can be repeated with more informative and up do date features.


## Inference
The inference pipeline first gets all unlabeled rows in "dataset.csv" that are not in the processed sets and stage them for inference. \
The current model and preprocessor get loaded, and rows get preprocessed and then passed to the model. \
Model output argmax gets added to the set of processed data, under the "PRED_TARGET" column.


## Monitoring
Monitoring gets all "dataset.csv" rows with "TARGET" and processed, with a "PRED_TARGET", defined as well and compares the two values to estimate the Average Accuracy. \
If the score goes below a configurable threshold RETRAIN_THRESHOLD, here set to 0.7, a state variable will signal model retraining. \
The results of monitoring get stored in a folder created ad hoc when the model was saved, under the "reports_mntr" folder. \
Results are visible as an HTML report that gets updated when monitoring runs. \
To prevent the first predictions of a model to weight too heavily on retraining triggering, a minimum size is needed for monitoring, configurable as MONITORING_WINDOW, here set to 5k.


## Training
There are 2 training pipelines:
- Training of a single specific model
- Training of all models in MODEL_REGISTRY

At start up time, all models training is signaled. \
When monitoring triggers retraining, the system performs the following in order, as the previous fails:
- Retrain the current model configuration on fresher data
- Retrain all models in MODEL_REGISTRY on fresher data
- Keep the system in an idle state, pausing inference, and send a notification email

In order to succeed, a model needs to score above a configurable DEPLOY_THRESHOLD, here set to 0.75 and suggested to be slightly higher than RETRAIN_THRESHOLD to prevent frequent retrainings. 

### Training Algorithm
The training procedure, first gets all "dataset.csv" rows with "TARGET", then filtering the top TEST_WINDOW (here 50k) and the following top TRAIN_WINDOW (here 200k), both configurable. \
To contain overfitting, training data get split in a configurable CV_FOLDS (here 3) number of folds, for a validation step. \
Each training fold get furtherly split in DRO_GROUPS folds over time (here 3), so to perform Group DRO: at each step one batch per fold gets evaluated and the gradient becomes the softmax of all gradient the batch of each group gave. \
Group DRO uses a loss that promotes a bounded loss across all groups, so here it promotes a representation that is robust over time. 

### Model Selection
Each model gets then retrained over all training data and the one with higher Average Accuracy over CV folds gets selected. \
Preprocessors get fit with each CV fold and overall data as well in the process. \
When a model gets deployed, it gets saved and its monitoring reports folder gets created below "reports_mntr". \
Both single model and all registry training create ad have reports, that get saved in per training folder created below "reports". \
Single training folder shows mean and variance of both training folds and validation folds. \
All registry training shows such plot for each model and one comprehensive model comparing mean validation mean and std at model selection. \
Results are visible in an HTML report as for monitoring.


## Pipelines and Orchestration
Actions performed by the system are combined in the following pipelines:
- Inference
- Monitoring
- All Registry Retraining
- Current Model Retraining
- Raw Tables Update

Pipelines are then orchestrated by "scheduler.py", that is the main script called by the Docker build. \
An APScheduler BlockingScheduler instance loops in a sleep-based waiting the following algorithm to execute them:
```
If Retrain is not required:
    - Update
    - Inference
    - Monitoring
otherwise:
    If Current Model Retraining has not been executed:
        - Current Model Retraining
    If All Registry Retraining has not been executed:
        - All Registry Retraining
```



## Log and Email Services
Results of the main actions performed on each pipeline get logged in a configurable log file. \
The only halting operation expected is all registry training failure against the expected DEPLOY_THRESHOLD, that put the system in an idle state. A human action is required, so sender and receiver emails con bi configured. Here the expectation is that some other models or hyperparameters configurations may be tried.


## Tests
"tests" folder contains the 3 subfolders "unit", "integration", "system" with the respective (19, 5 and 5) tests. \
They tests all code but for "scheduler.py" and "docker/Dockerfile", even if they try to simulate all possible code flows.


## Possible Improvements
The project has the goal of showing understanding of theoretical and MLOps concepts. Possible improvements are:
- Monitoring not only model performance but also univariate and covariate shifts.
- Extend static training with meta-optimization, either based on Genetic Algorithms or Bayesian Optimization
- Make a configurable set of collectable metrics, for optimization or reporting purposes
- Add a per model preprocessing in between data dependent existing preprocessing and models, with a PREPROCESSING_REGISTRY to reference in MODEL_REGISTRY configurations
- Add a proper interface to expose a web API to make the model a Web Service


## References
1. Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.