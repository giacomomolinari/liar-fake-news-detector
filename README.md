# LIAR Fake News Detector

## Overview
This is an end-to-end NLP project that critically evaluates the LIAR benchmark for fake news detection. The project includes methodological analysis and multiple model architectures, from classical ML baselines to fine-tuned BERT. It is a learning project, developed while working through Hands-On Machine Learning (Géron) and Pattern Recognition and Machine Learning (Bishop). 

The LIAR benchmark was introduced by [Wang (2017)](https://arxiv.org/abs/1705.00648). The dataset contains short statements given by (mostly) American political and cultural figures in various contexts, taken from fact-checking website [PolitiFact](https://www.politifact.com/). On top of the statements, the dataset contains metadata about the topic and speaker, as well as a short description of the context for each statement. Statements are labeled as either "true", "mostly true", "half true", "mostly false", "false", and "pants on fire". 

## Key Findings

### 1. Label Validity 
I argue that the `pants-on-fire` category is potentially problematic. It makes the labels asymmetric (half-true is not halfway between the most and least truthful label) and could be encoding some facts about whether the statement is considered obscene, extraordinary, or just surprising, which are arguably not what a lie-detector should be trained to recognize. Therefore, I will merge the `pants-on-fire` category with the `false` category in this project.

### 2. Data Leakage
The dataset includes a set of `speaker history` features for each statement, which provide a total count of the speaker's statements for each truthfulness label in the dataset, across both train, validation, and test sets. For example, if the sample is a false statement by Mitt Romney, the HistFalse feature is the number of False statements by Mitt Romney in the dataset, **including the current one**.

This raises serious data leakage concerns: the `speaker history` features of each sample contain both information about that sample's label, and information about frequency of labels for that sample's speaker in both the validation and test datasets, so they should not be passed to the models.

### 3. Model Performance
Due to my merging of the `pants fire` label into the `false` label, the `false` label ends up being much more frequent than all the others in the dataset. I show that this leads to most models collapsing into majority-class classifiers during training.

To mitigate this problem, I train models using a rebalanced dataset. On this balanced dataset, I show that metadata-only models (i.e. models trained on the metadata, but not on the actual statements) can improve on the majority-class classifier's accuracy by 6-8 percentage points. Models using BERT pretrained embeddings + RNN improve on the majority-class classifier's accuracy by up to 12 percentage points, reaching 30-32% accuracy on the balanced 5-class problems, where a majority-class classifier has only 20% accuracy. 

None of the feature combinations I explore shows a significant or consistent advantage over the text-only BERT embeddings model. 
And interestingly, more advanced models using BERT's contextualised embeddings, or even a full BERT model, do slightly worse than the model using BERT's pretrained embeddings + RNN. This may be due to the dataset being relatively small, especially after rebalancing, so that the validation data may not be able to reliably capture differences of 2-3% generalization accuracy. 

## Project Structure
The project consists of two notebooks:
- `eda.ipynb` contains some exploratory data analysis of the LIAR dataset. Several pre-processing decisions, including the merging of `pants-fire` and `false` labels, are motivated in this notebook.
- `models.ipynb` contains the models with their training and fine-tuning code. After running Sections 1 and 2 (getting the data + basic preprocessing) you should be able to run any of the other sections independently on the rest. 

The project was developed and is intended to be run on Google Colab. Package installation is handled within the notebooks

## Dataset
You can download the dataset from [William Yang Wang's website](https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip). Both notebooks assume that a copy of the dataset is stored in your Google Drive. You can specify the path to your copy of the dataset by modifying the `DATASET_PATH` variable at the start of each notebook.

## Running the Notebooks
To run the notebooks, you can just click the "Open in Colab" in the notebooks themselves. Alternatively you can use the links below:
- [eda.ipynb](https://colab.research.google.com/github/giacomomolinari/liar-fake-news-detector/blob/main/eda.ipynb)
- [models.ipynb](https://colab.research.google.com/github/giacomomolinari/liar-fake-news-detector/blob/main/models.ipynb)
Note that a Google Drive mount is required for the dataset. Both notebooks include the code to mount the Google Drive, but you will need to authorize the mount when running them.

## Limitations
There are several limitations for this project:
1. The benchmark samples statements from the political fact-checking website [PolitiFact](https://www.politifact.com/). This means that the statements are sampled by the website's editorial choices. For example, a fact-checking website may focus on statements that make the news more than on less controversial statements, or statements on more ordinary subjects. Secondly, it's possible that the website may have a political or personal bias.
2. When using pretrained BERT models, there is some risk of contamination, since BERT was trained (among other things) on English Wikipedia up to 2018. Since some of the statements in the corpus are well-known statements by famous political figures, and they are all pre-2018, it is possible that they may appear in a Wikipedia page which mentions whether the statement is true or not. 

## License
This project is licensed under the [MIT License](https://github.com/giacomomolinari/liar-fake-news-detector/blob/main/LICENSE).
