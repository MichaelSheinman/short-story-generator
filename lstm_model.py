# Importing everything from clean.py to get the formatted. clean data
from clean import *
import torch
import torch.nn as nn
import time
import numpy as np
import math
import matplotlib.pyplot as plt

# THIS IS THE CODE FOR THE TITLE GENERATION MODEL----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# --------------------------- 1. FORMAT THE TITLES --------------------------------
# Lowercase, remove punctuation and numbers from titles
def clean_title(title):
    '''
    Removes punctuation, lowercases and numbers from titles
    '''
    # upper- to lowercase
    title = str(title).lower()

    # remove numbers
    title = re.sub(r"[0123456789]+\ *", " ", title)

    # remove punctuation
    title = re.sub(r"[,.&$%<>@#?-_*/\()~='+;!:`]+\ *", " ", title)
    title = re.sub("''", ' ', title)
    title = re.sub('-', ' ', title)

    # remove duplicated spaces
    title = re.sub(' +', ' ', title)
    return title.strip()

# --------------------------- Creating the vocabulary ----------------------------
end_of_sentence = '.' # symbol to denote the end of the sentence
def create_vocabulary(stories):
    '''
    Creates a vocabulary of the story titles.
    '''
    vocabulary = set()
    for title in stories:
      title_words = clean_title(title).split(" ")
      vocabulary.update(title_words)
    word_list = list(vocabulary)
    word_list.append(end_of_sentence)
    vocabulary = {word_list[word]:word for word in range(0,len(word_list))}
    return vocabulary

# create vocabulary of the fairy tale titles
# titles: index for the one-hot shot
VOCABULARY = create_vocabulary(STORIES)
vocab_size = len(VOCABULARY)

print("The vocabulary:")
print(VOCABULARY)

print(f"Total number of unique words: {vocab_size}")


# --------------------------- 2. PREPARE THE TRAINING SET --------------------------
# Word to tensor encodings ...

# Translate word to an index from vocabulary
def word_to_index(word):
    if (word != end_of_sentence):
        word = clean_title(word)
    return VOCABULARY[word]

# Translate word to 1-hot tensor
def word_to_tensor(word):
    tensor = torch.zeros(1, 1, vocab_size)
    tensor[0][0][word_to_index(word)] = 1
    return tensor

# Turn a title into a <title_length x 1 x vocab_size>,
# or an array of one-hot vectors
def title_to_tensor(title):
    title_words = clean_title(title).split(' ')
    tensor = torch.zeros(len(title_words) + 1, 1, vocab_size)
    for index in range(len(title_words)):
        tensor[index][0][word_to_index(title_words[index])] = 1

    tensor[len(title_words)][0][VOCABULARY[end_of_sentence]] = 1
    return tensor

# Turn a sequence of words from title into tensor <sequence_length x 1 x vocab_size>
def sequence_to_tensor(sequence):
    tensor = torch.zeros(len(sequence), 1, vocab_size)
    for index in range(len(sequence)):
        tensor[index][0][word_to_index(sequence[index])] = 1
    return tensor

# ----------------------------- 3. CREATING THE SEQUENCES --------------------------------
# Generate sequences out of titles:
sequence_length = 4 # hyperparam that can be tweaked

# Generate sequences
def generate_sequences(stories):
    sequences = []
    targets = []
    # Loop for all selected titles
    for title in STORIES:
        # Run through each title
        if clean_title(title) != '' and clean_title(title) != ' ':
            title_words = clean_title(title).split(' ')
            #print(title_words)
            title_words.append(end_of_sentence)

            for i in range(0, len(title_words) - sequence_length):
                sequence = title_words[i:i + sequence_length]
                target = title_words[i + sequence_length:i + sequence_length + 1]

                sequence_tensor = sequence_to_tensor(sequence)
                target_tensor = sequence_to_tensor(target)

                sequences.append(sequence_tensor)
                targets.append( target_tensor)

    return sequences, targets
# generate sequences for all the story titles!
sequences, targets = generate_sequences(STORIES)


# ----------------------------- 4. LSTM MODEL STRUCTURE --------------------------------
class LSTM_model(nn.Module):
    '''
    Simple LSTM model to generate bedtime story titles.
    Arguments:
        - input_size - should be equal to the vocabulary size
        - output_size - should be equal to the vocabulary size
        - hidden_size - hyperparameter, size of the hidden state of LSTM.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_model, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.view(1, 1, -1), hidden)
        output = self.linear(output[-1].view(1, -1))
        output = self.softmax(output)
        return output, hidden

    # the initialization of the hidden state
    # using cuda speeds up the computation
    def initHidden(self, device):
        return (torch.zeros(1, 1, num_hidden).to(device), torch.zeros(1, 1, num_hidden).to(device))


# ---------- Initialize LSTM

num_hidden = 128 # number of hidden units , hyperparam

# inputs and outputs of RNN are tensors representing words from the vocabulary
rnn = LSTM_model(vocab_size, num_hidden, vocab_size)

#  Function to convert the output of the model into a word
def output_to_word(output):
    '''
    Return the index from the vocabulary and the corresponding word
    '''
    top_n, top_i = output.topk(1)
    index_category = top_i[0].item()
    return [key for (key, value) in VOCABULARY.items() if value == index_category], index_category

# Function that translates an index of the word in the vocabulary into tensor
def tensor_to_index(target):
    '''
    Return the tensor containing target index given tensor representing target word
    '''
    top_n, top_i = target.topk(1)
    target_index = top_i[0].item()
    target_index_tensor = torch.zeros((1), dtype = torch.long)
    target_index_tensor[0] = target_index
    return target_index_tensor

# ----------------------------- 5. TRAINING THE LSTM MODEL  --------------------------------
learning_rate = 0.005
criterion = nn.NLLLoss()

# device to use: using cuda if it's available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Define training procedure
def train(sequence, target, device):
    # Move tensors to device
    hidden = rnn.initHidden(device)
    sequence = sequence.to(device)
    target = target.to(device)

    rnn.zero_grad()

    # Forward step
    for i in range(sequence.size()[0]):
        output, hidden = rnn(sequence[i], hidden)

    output, hidden = rnn(sequence[i], hidden)

    loss = criterion(output, tensor_to_index(target).to(device))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

# The actual training and get the model's training curve
# Set up the number of iterations, printing and plotting options
n_iters = 1100000
print_every = 1000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

rnn = rnn.to(device)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Shuffle indices
indices = np.random.permutation(len(sequences))

start = time.time()

# Run training procedure
for iter in range(1, n_iters + 1):
    # Pick index
    index = indices[iter % len(sequences)]
    # Run one training step
    output, loss = train(sequences[index], targets[index][0].long(), device)
    current_loss += loss

    # Print iter number and loss
    if iter % print_every == 0:
        guess, guess_i = output_to_word(output)
        print('%d %d%% (%s) Loss: %.4f' % (iter, iter / n_iters * 100, timeSince(start), loss))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
# Plot training loss
plt.figure()
plt.plot(all_losses)


# Generates title given the first word
def generate_title(first_word):
    max_num_words = 5 # in a title
    sentence = [first_word]

    # Initialize input step and hidden state
    input_tensor = word_to_tensor(first_word)
    hidden = (torch.zeros(1, 1, num_hidden).to(device), torch.zeros(1, 1, num_hidden).to(device))
    output_word = None
    i = 1

    # Generate title
    while output_word != '.' and i < max_num_words:
        input_tensor = input_tensor.to(device)
        output, next_hidden = rnn(input_tensor[0], hidden)
        final_output = output.clone().to(device)

        # Use the probabilities from the output to choose the next word
        probabilities = final_output.softmax(dim=1).detach().cpu().numpy().ravel()
        word_index = np.random.choice(range(vocab_size), p = probabilities)

        output_word = [key for (key, value) in VOCABULARY.items() if value == word_index][0]
        sentence.append(output_word)

        # update
        input_tensor = word_to_tensor(output_word)
        hidden = next_hidden
        i += 1

    if sentence[-1] != ".": sentence.append(".")

    return sentence

# ----------------------------- 5. TESTING LSTM MODEL OUTPUTS --------------------------------
# Testing on different iterations of the title with the same word start
num_titles = 10
print(f"Generating {num_titles} titles...\n")
for i in range(num_titles):
    sampled_title = generate_title("white") # plug in word from the VOCABULARY
    title = ' '.join(sampled_title)
    print(title)



