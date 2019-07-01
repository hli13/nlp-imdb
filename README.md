# Sentiment Analysis of IMDb Reviews using NLP techniques

In this project, Natural Language Processing (NLP) models are trained to detect the sentiment (positive or negative) of movie reviews. The data is preprocessed and tokenized with the help of [NLTK](https://www.nltk.org/) python package. Both the bag-of-words and language models are trained. The Recurrent Neural Network (RNN) with long short-term memory (LSTM) layers is used to incorporate the temporal infomation of the text. In addition, the trained language model is used to generate fake reviews with customizable sentiments.

## Dependencies

```
numpy==1.16.4
matplotlib==3.1.0
torch==0.4.1
torchvision==0.2.1
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
curl  -o aclImdb.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```
Then, extract the file with
```
tar -xzf aclImdb.tar.gz
```

## Result

TBD
