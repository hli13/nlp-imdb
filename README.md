# Sentiment Analysis of IMDb Reviews using NLP techniques

> Code refactoring is in progress

In this project, Natural Language Processing (NLP) models are trained to detect the sentiment (positive or negative) of movie reviews. The data is preprocessed and tokenized with the help of [NLTK](https://www.nltk.org/) python package. Both the bag-of-words and language models are trained. The Recurrent Neural Network (RNN) with long short-term memory (LSTM) layers is used to incorporate the temporal infomation of the text. In addition, the trained language model is used to generate fake reviews with customizable sentiments.

## Dependencies

```
numpy==1.16.4
matplotlib==3.1.0
nltk==3.4.4
torch==1.1.0
torchvision==0.3.0
```

## Dataset

The [Large Movie Review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) is used to train and evaluate the NLP models. It contains 25,000 labeled higly polar movie reviews and 25,000 unlabled reviews for testing.

## Implementation

TBD

## Hyerparameters

TBD

## Running the model

Run the following script to make sure required packages are installed.
```
pip install -r requirements.txt
```
Download the Large Movie Review dataset using the following command.
```
wget -O aclImdb.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```
or
```
curl -o aclImdb.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```
Then, extract the file with
```
tar -xzf aclImdb.tar.gz
```
Download [GloVe](https://nlp.stanford.edu/projects/glove/) features with the following script.
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
```
Then, extract the file
```
tar -xzf glove.840B.300d.zip
```

## Result

TBD
