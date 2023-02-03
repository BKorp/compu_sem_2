# 1. compu_sem_2

## 1.1. General information
This repository contains the code that was used for a [University of Groningen](https://www.rug.nl/?lang=en) research project for the course [computational semantics](https://ocasys.rug.nl/current/catalog/course/LIX021M05).

It can be roughly divided into the `BERT-based models`, which can be found in the `root` directory, and the `T5 models`, which can be found in the so-named `T5` folder.

The data that was used for the project was taken from [The Parallel Meaning Bank (PMB)](https://pmb.let.rug.nl/data.php). The specific data that was used was the data from `Discourse Representation Structures` version `4.0.0`. We used both a setup containing only the `English Gold labels` and a setup containing `all of the labels (English, German, Italian, and Dutch)`.

Requirements to run the project can be found in `requirements.txt`.

## 1.2. The BERT-based models

To run BERT (bert-base-uncased), RoBERTa (roberta-base), or BERT multilingual (bert-base-multilingual-uncased), download the BERT_Semantic_Tagger.ipyn notebook file. This notebook only takes the PMB data file as input, which can be either of the two in the data folder (1. 'sem-pmb_4_0_0-gold.csv', 2. 'sem-pmb_4_0_0-all-gold.csv'). The first file is only the English PMB data, which can be used with either BERT or RoBERTa, the second file is multilingual PMB data and can only be used with BERT multilingual.

Each model has already been run with our data, and the results are displayed in the output files. To re-run the notebook, please use a Jupyter Notebook or Google Colab environment or convert the notebook to a .py format to run with your own preferred environment. 

## 1.3. The T5 models

### 1.3.1. Running the models
Once the requirements have been fulfilled, each model can be run using the terminal for .py and .pyz programs, all other programs are notebooks.

For the terminal programs, it is recommended that one familiarizes themselves with the possible commands through the help function:
```bash
python $PROGRAM -h
```

### 1.3.2. Initial_setup
The so-called `initial` setup consists of an encoding for t5 input tokens and output tags that splits output tags into the following format:
```
Input semtag: We went fishing in the lake .
----------------------------------------------------------------
Output PRO: We ; PST: went ; EXG: fishing ; REL: in ; DEF: the ; CON: lake ; NIL: .
```
The folder structure consists of two basic steps:

1. `training`
2. `Generation`

During training, a T5 model is finetuned on given train data, after which the input is moved (manually) to the generation step, where the finetuned model is used to generate the outputs for the test set.

After the generation, the outputs are compared to the expected outputs of the test set, and a classification report is made.

### 1.3.3. Sentinel_setup
For further testing, another setup was created using a `sentinel` setup:
```
Input semtag: 〈extra_id_0〉 We 〈extra_id_1〉 went 〈extra_id_2〉 fishing ....
----------------------------------------------------------------
Output 〈extra_id_0〉 PRO 〈extra_id_1〉 PST 〈extra_id_2〉 EXG ....
```
Here the `extra_id`'s are used to keep track of the different translations for the input and output.

The folder structure consists of three basic steps:

1. `training`
2. `Generation`
3. `Output testing`

During training, a T5 model is finetuned on given train data, after which the input is moved (manually) to the generation step, where the finetuned model is used to generate the outputs for the test set.

After the generation, the outputs are compared to the expected outputs of the test set, and a classification report is made.

The difference here is that the outputs of the generation step are first stored before they are then compared in step 3.
