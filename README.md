<h1>Bedtime Story Completion Model</h1>
<h3>CSC413 - Neural Networks & Deep Learning Final Project</h3>

## Introduction

We created two models to aid writers with creating short stories.

- A short-story generator: A model that takes the start of a short story and continues the story.
- A title generator: a sequence-to-sequence model that generates a title given a short story.

The two models can be used together in an application to help writers overcome the writer's block and improve their speed of writing short stories.

For the short-story generation, we fine-tuned a pre-trained GPT-2 model.
For the title generation, we trained an RNN model from scratch. Both were trained on a dataset comprised of childrens' books. Therefore, we present our model in the context of generating this type of story --- though we believe that the architecture employed should be similarly useful for other genres.

## Model Figure

#### GPT2 model:

GPT-2 is primarily comprised of a sequence of transformer blocks, each employing (masked) multi-headed attention ([Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)), with multiple normalization layers at the beginning of each block, plus one normalization layer at the end of the transformer stack ([Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)). The exact number and size of these blocks depends on the size of model used (see "Model Parameters" below). Note the GPT architecture is decoder-only.

Inputs are represented as tokens, though the relationship between words and tokens is not one-to-one (see "Data Transformation" below for more details). Of note, the transformer sequence is preceded by an encoder which encodes these tokens and applies positional encoding. When carrying out text generation, the transfomer sequence is followed by a linear modelling head, which is a "linear layer with weights tied to the input embeddings" ([Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2))

#### RNN model:

RNN with LSTM cell can be broken down into three main components: the input layer, the LSTM layer, and the output layer. The LSTM layer consists of a series of LSTM cells that process the input sequence in a sequential manner. Each LSTM cell has three main components: the input gate, the forget gate, and the output gate. These gates control the flow of information in and out of the cell, allowing the LSTM to selectively remember or forget information from previous time steps. The output layer of an RNN with LSTM cell receives the final hidden state output from the LSTM layer and produces the final output for the sequence.

## Model Parameters

#### GPT2 Model:

Due to compute limitations, we present our results based on our use of the smallest GPT2 instance, which has approximately 117 \* 1024^2 parameters (117M) ([Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)), laid out across 12 layers (i.e. 12 transformer blocks), whose inputs and outputs are 768-dimensional. The attention layers work with 1024 token context lengths and have 12 heads. The feed-forward layers work with dimension 3072 (as in GPT1).

In particular, we have the following parameter counts:

For the input embedding layer, we have 38597376 parameters. This corresponds to 50257, the vocab size, times 768, the embedding dimensionality.

For the positional embedding, we have 786432 parameters. This corresponds to 1024, the positional embedding size, times 768, the embedding dimensionality.

For the transformer stack, we have 12 transfomer layers, each with one self-attention layer, one feed-forward layer, and one normalization layer:

- The normalization layer has a relatively negligible 1536 parameters (twice the embedding dimension for the two distribution parameters each).
- For the self-attention layer, we have approximately 4 \* (768^2) parameters, corresponding to a linear output (768^2 matrix), and multiple Q, K, V matrices (12 heads \* 3 matrices each in total, each matrix having 768 \* (768/12) parameters --- notice here that the outputs for these is divided by the head count, which ensures concatenating the attentions will result in a 768-dimensional vector). Refere to the source below for more details (and the exact formula, if desired).
- For the feed-forward layer, there are two linear layers that start with the 768-dimensional embedding, grow it to 3072 dimensions, apply ReLU, and then bring it back to a 768-dimensional embedding. This corresponds to a total parameter count of (768 \* 3072 + 3072) + (768 \* 3072 + 768) (matrix weights and biases).

In total this yields approximately `4 * (768^2) + 1536 + (768 * 3072 + 3072) + (768 * 3072 + 768)` parameters per layer times 12 layers, or almost 85 million parameters --- not the exact value of 85054464 since we neglected a few biases in our calculation above for simplicity --- but very close.

(See [How to Estimate the Number of Parameters in Transformer models](https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0) for details on this calculation)

The last normalization layer `ln_f` has a relatively negligible 1536 parameters (which is two times the embedding dimension)

Note that the language modelling head has weights tied to the input embedding, so it does not add to our parameter count.

#### RNN Model:

## Model Examples

#### GPT2 Model:

#### RNN Model:

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

#### GPT2 Model:

#### RNN Model:

Below is the training loss curve generated in lstm_model.py. The plot was generated by conducting 1100000 iterations of training and graphing the resulting loss at intervals of 1000 iterations. The hyperparameters used for training the model were as follows:

- `learning_rate` = 0.005
- `n_hidden` = 128
- `sequence_length` = 4

<img src='images/train_curve.png' width="40%" height="40%">

## Hyperparameter Tuning

#### GPT2 Model:

#### RNN Model:

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

For the GPT-2 model we investigated a few different quantitative methods of measuring our model, including ROUGE, BERTScore, and BLEU. We decided to used the BLEU score as our measure as we found it to be the most widespread, simple, and effective metric for our task. BLEU stands BiLingual Evaluation Understudy. It's a quantative metric to evaluate the quality of translated text. The metric was first proposed in a 2002 [paper](https://aclanthology.org/P02-1040.pdf?ref=blog.paperspace.com) for automatic evaluation of machine translation.

BLEU evaluates the quality based on the similarity of two texts. It provides a score between 0 and 1 to indicate how close the text is a human text. We implement this method by cutting off the end of a generated story and using our model to generate the continuation of the text.

## Results
#### GPT2 Model:

#### RNN Model:

In the case of the title generation we get the following sampled titles with a sequence_length = 3:  
```
indian dickon tea .
penny maleen .
norroway paris .
herdsman discontented .
real are .
year blackfeet cudred on how the how all his his pipes saw
are anything to seeing food wife xi white hump cobbler’s learn learn
anne cudred .
tuk .
godmother ix .
goat imagine criticism .
chap couch .
claus to peace strife d london george .
tobacco wen .
bad jolly on are were x all how in in sorrow tempests
```
We get the following sampled titles with a sequence_length = 4:  

**Sample 1**  
```
riding among war .
allowed fable .
scuttle stag .
rest shuttle .
closing moore iv .
know when hood your question ruler oh out and began borrow began
walk tash .
santa bear .
leaves hill .
stork march .
sack girl .
angels .
vanity leyden .
beeches ode iv .
midsummer .
```

**Sample 2**
```
mate sparrow .
strangest frost .
stolen xii .
much huntsman actors .
publishers midsummer .
jackdaw sir .
proserpine knowall lip .
harm maid .
mole fine .
child heidelberg .
teeny frank .
clever delight .
inferior coronet history .
feeler luck .
meets iv .
```
We get the following sampled titles with a sequence_length = 5:  

**Sample 1**
```
show fed pack himself himself .
hogshead cyclone n debarred discovered .
period orphant magic world .
balloon appears lead himself .
boscombe tail r s wen .
youth right he command was who if the the get ready one’s
vampire .
balloon understand thought peace pieces .
ball kindness experiments nail discovered .
u tobacco .
chorus slipper .
rowland woodman king’s do readily .
harpies beautiful pilgrims tears’ .
prevented guard do under himself .
play shadow ramhah .
```
**Sample 2**
```
bye power .
age seasons witch publishers .
right river witch was he he not not not not not not
remain belt .
famine wives secret .
period punch .
jack bennett .
impressions out chapter i escape .
suds miacca .
happy kingfisher .
mountain kangaroo away other oldtown .
runs bright wait vi himself .
china coffin jewels he know must .
plain brook tobacco beth .
eggs land childish readily escape i .
```
## Justification of Results

One of the big challenges we ran into was simply compute. Fine tuning GPT2 is very computationally expensive. Additionally, when we attempted to use GPT2 versions with large parameter counts, we found that we quickly ran out of memory. Therefore, we had to stick with a smaller parameter count, which
reduces the quality of our outputs.

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

- Contributed to code for finetuning GPT2 and employing GPT2 for story completion.
- Implemented validation for finetuned GPT2.
- Worked on parts of the writeup.
- Carried out exploratory work on some model architectures, including direct integration of finetuned GPT2 with RNN (unfortunately I found that this seemed to be computationally infeasible).
- General contributions to debugging.

Michael Sheinman:

- Finetuned GPT-2 and used GPT-2 for story completion. Completed initial overfitting on a single data point, training, checkpoints, and testing.
- Worked on parts of the writeup.
- Combined the GPT-2 and RNN models into a single file by uploading both checkpoint files externally to dropbox.
- Attempted to setup development using the lab machines GPU. Unfortunately, I was not able to accomplish this due to a lack of installation permissions, issues with Anaconda virtual environments, and no reply from Andrew Wang.
- Attempted setting up 3 different architectures: NanoGPT, GPT-2 Medium, and plain transformers. Found that none of these architectures were feasible due to Google Collab limitations and short story sequences being too long.

Ekaterina Semyanovskaya:
