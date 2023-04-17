<h1>Bedtime Story Completion Model</h1>
<h3>CSC413 - Neural Networks & Deep Learning Final Project</h3>



## Introduction
We created two models to aid writers with creating short story.   

* A short-story generator:  A model that takes the start of a short story and continues the story. 
* A title generator: a sequence-to-sequence model that generates a title given a short story. 

The two models can be used together in a web-application to help writers overcome the writer's block and improve their speed of writing short stories. 


## Model Figure
GPT2 model:



RNN model:



## Model Parameters 
GPT2 Model:

RNN Model: 

## Model Examples 


## Data  
Data source: https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus

## Data Transformations

For the story completion model, we use a pre-trained GPT2 text encoder, which encodes the text using Byte-Pair Encoding (BPE).

For the title generator, we use a naive bag-of-words encoder. This has the limitation that the model cannot produce titles using words not seen in the training set.

## Data Split 

## Training Curve

## Hyperparameter Tuning

## Quantitative Measures

## Results

## Justification of Results



## Ethical Considerations
<br>

## Authors
<br>