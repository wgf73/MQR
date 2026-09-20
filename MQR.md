# MQR: Multi-Granularity Query Routing for Medication Recommendation

## Installation

1. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested.

   ```
   conda create --name MQR
   ```

2. Activate the newly created environment.

   ```
   conda activate MQR
   ```

## Download the data

1.  You must have obtained access to [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) databases before running the code.
2.  Download the MIMIC-III and MIMIC-IV datasets, then unzip and put them in the `data/input/` directory. Specifically, you need to download the following files from MIMIC-III: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`, and the following files from MIMIC-IV: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`.
3.  Download the [drugbank_drugs_info.csv](https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing) and [drug-DDI.csv](https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing) files, and put them in the `data/input/` directory.

## Process the data

Run the following command to process the data:

```
python process.py
```

## Run the models

~~~bash
python main_MQR.py -dataset mimic-iii
python main_MQR.py -dataset mimic-iv 
~~~

Welcome to contact me [wanggongfu.qlu@foxmail.com] for any question.

