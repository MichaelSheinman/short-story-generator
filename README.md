<h1>Bedtime Story Completion Model</h1>
<h3>CSC413 - Neural Networks & Deep Learning Final Project</h3>

## Introduction

We created two models to aid writers with creating short story.

- A short-story generator: A model that takes the start of a short story and continues the story.
- A title generator: a sequence-to-sequence model that generates a title given a short story.

The two models can be used together in an application to help writers overcome the writer's block and improve their speed of writing short stories.

For the short-story generation, we fine-tuned a pre-trained GPT-2 model.
For the title generation, we trained an RNN model from scratch.

## Model Figure

GPT2 model:

GPT-2 is primarily comprised of a sequence of transformer blocks, each employing multi-headed attention (Improving Language Understanding by Generative Pre-Training), with multiple normalization layers (Language Models are Unsupervised Multitask Learners). Inputs are represented as tokens, though the relationship between words and tokens is not one-to-one.

RNN model:

RNN with LSTM cell can be broken down into three main components: the input layer, the LSTM layer, and the output layer. The LSTM layer consists of a series of LSTM cells that process the input sequence in a sequential manner. Each LSTM cell has three main components: the input gate, the forget gate, and the output gate. These gates control the flow of information in and out of the cell, allowing the LSTM to selectively remember or forget information from previous time steps. The output layer of an RNN with LSTM cell receives the final hidden state output from the LSTM layer and produces the final output for the sequence.

## Model Parameters

GPT2 Model:

RNN Model:

## Model Examples

GPT2 Model:

RNN Model:

Successful story title generated: "teeny frank ."  
Unsuccessful story title generated: "year blackfeet cudred on how the how all his his pipes saw ."

The first title is considered successful since it is concise, logical and descriptive. It was generated with the `sequence_length = 4` hyperparameter.  
The second title is considered unsuccessful since it lacks meaning. It was generated with the `sequence_length = 3` hyperparameter.

## Data

We used the children stories text corpus data set, available on https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus. This data set is a collection of children books collected from Project Gutenberg. Project Gutenberg aims to encourage the distribution of eBooks by sharing books or stories that are in the public domain, meaning there are no intellectual property right over the data. The Kaggle dataset is similarly under the public domain (Creative Commons license) waiving rights to copyright law. Hence we have permission to use the data.

Since the data is a collection of short stories, it is ideal for text generation. The dataset consists of 20.6 MB of English text. This is sufficient for our purposes as we are starting with the GPT-2 model, so we only need to fine-tune the model for the dataset. As for title generation, the dataset contains the titles of all the stories, so we can use that for training and evaluation.

The dataset consists of around 1000 stories, with an average story length of around 3014 words. There are a total of 3,792,498 words, with 151,964 unique words. There are 19,304 question marks, 23,270 exclamation marks, 179,871 periods, and 309,764 commas. The top 5 most common words along with their frequencies are "the" (209,801), "and" (139,615), "to" (99,378), "of" (85,344), and 'a' (79,847).

Snippet of a short story in the dataset. Overall length is 411 words:

    THE REAL PRINCESS
    There was once a Prince who wished to marry a Princess; but then she must be a real Princess. He travelled all over the world in hopes of finding such a lady; \textbf{[...]} The Prince accordingly made her his wife; being now convinced that he had found a real Princess. The three peas were however put into the cabinet of curiosities, where they are still to be seen, provided they are not lost.
    Wasn't this a lady of real delicacy?

## Data Transformations

The initial data consisted of a single text file where the titles and stories were not distinctly separated. To make the data usable for our project objectives, we required it to be organized in a structured manner. Consequently, we converted the text file data into a dictionary format, where the titles served as the keys and the stories as the corresponding values. Please refer to the "clean_data" function for further details.

For the story completion model, we use a pre-trained GPT2 text encoder, which encodes the text using Byte-Pair Encoding (BPE). We also restrict the length of the resulting sequence,
given that GPT2 has a limited context length, which is possibly exceeded by some of the
stories.

For the title generator, we use a naive bag-of-words encoder. This has the limitation that the model cannot produce titles using words not seen in the training set.

## Data Split

## Training Curve

GPT2 Model:

RNN Model:

Below is the training loss curve generated in lstm_model.py. The plot was generated by conducting 1100000 iterations of training and graphing the resulting loss at intervals of 1000 iterations. The hyperparameters used for training the model were as follows:

- `learning_rate` = 0.005
- `n_hidden` = 128
- `sequence_length` = 4

<img src='images/train_curve.png' width="40%" height="40%">

## Hyperparameter Tuning

GPT2 Model:

RNN Model:

1. Tune `sequence_length`.

The length of the sequence is a hyperparameter. The model was trained with the three values of `sequence_length`:

- `sequence_length = 3`
  Sample titles output:

```
real are .
year blackfeet cudred on how the how all his his pipes saw
are anything to seeing food wife xi white hump cobbler’s learn learn
tuk .
godmother ix .
```

- `sequence_length = 4`
  Sample titles output:

```
teeny frank .
strangest frost .
stolen xii .
publishers midsummer .
clever delight .
```

- `sequence_length = 5`
  Sample titles output:

```
show fed pack himself himself .
hogshead cyclone n debarred discovered .
period orphant magic world .
balloon appears lead himself .
boscombe tail r s wen .
```

From the analysis of multiple output blocks, `sequence_length = 4` is the better hyperparameter choice.

2. Tune `learning_rate`.

The learning rate is a hyperparameter. The model was trained with the two values of `learning_rate`:

- `learning_rate = 0.05`
  Sample titles output:

```
.
midsummer gideon woman huslo awl awl the .
mount .
red .
.
```

- `learning_rate = 0.005`
  Sample titles output:

```
teeny frank .
strangest frost .
stolen xii .
publishers midsummer .
clever delight .
```

From the analysis of multiple output blocks, `learning_rate = 0.005` is the better hyperparameter choice.

## Quantitative Measures

## Results

## Justification of Results

One of the big challenges we ran into was simply compute. Fine tuning GPT2 is very computationally expensive.

Additionally, we noticed that validation loss tended to level off after a small number of epochs. Given that modern LLMs are usually trained with relatively few passes over the data, this is not completely unexpected. Even so, it may indicate limitations in our training process.

## Ethical Considerations

The ethical implications of our model are common to any text generation model.
Firstly, there is no guarantee that the output of the model is unique. In case of overfitting, the output data of the model may be identical to the contents of the dataset and even if the model does not reproduce the actual content of these stories, it may still copy the authors' style, ideas, or other valuable details. Another implication is the issue of intellectual property and rights. The story generated by AI raises questions about who owns the intellectual property rights to the generated content. If a machine creates a piece of content, should it belong to the person who trained the machine, the machine itself, or no one?

It is also important to notice that despite having 20.6 MB of data, the stories of the dataset themselves might not be inclusive or diverse enough for children of different backgrounds. This is an important issue because bedtime stories can give children perspectives on the world and representation at a young age is important for a child's development. Depending on the timeline when the stories of the dataset were written, there may be a lot of detrimental bias in the stories that portray certain groups of people as villanous and others as heroic. On that note, the model could potentially generate harmful content since there is no human supervision of the generated text. This leads us to the implication of responsibility and accountability. The stories generated by AI raise questions about who is responsible for the content generated by the machine in case of harmful output. Should the responsibility fall on the developer, the user, or the machine itself?

## Authors

<br>
Laura Maldonado :
- Worked on cleaning and pre-processing the dataset
- Setting up the LSTM model
- Putting the code for the models on the repo
- Implementing final.py: where the user interacts with the project

Mateus Moreira :

Michael Sheinman:

Ekaterina Semyanovskaya:
