# Importing everything from clean.py to get the formatted. clean data
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from clean import *

# Other imports for the model
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device_name = torch.device("cuda")
else:
    device_name = torch.device("cpu")
print(f"Using {device_name}")

# THIS IS THE CODE FOR THE STORY GENERATION MODEL----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# --------------------------- 1. SPLIT THE DATA  --------------------------------
# Due to GPU constraints, we cannot train/validate on a large number of data points.
# As such, we split as follows:

training_set_size = int(total_data * 0.7)
validation_set_size = int(total_data * 0.1)
testing_set_size = int(total_data * 0.2)

random.seed(2)   # ensuring the sets are consistent through different runs
random.shuffle(sorted_stories)

training_set = sorted_stories[:training_set_size]
validation_set = sorted_stories[training_set_size:
                                training_set_size + validation_set_size]
testing_set = sorted_stories[training_set_size + validation_set_size:]

# --------------------------- 2. TOKENIZE THE INPUT --------------------------------
# Dataset comes from torch.utils.data


class Stories(Dataset):
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.stories = []

        for story in control_code:
            self.stories.append(torch.tensor(
                self.tokenizer.encode(f"<|startoftext|>{story}<|endoftext|>")[
                    :max_length]
            ))
        if truncate:
            self.stories = self.stories[:20000]
        self.stories_count = len(self.stories)

    def __len__(self):
        return self.stories_count

    def __getitem__(self, item):
        return self.stories[item]


dataset = Stories(training_set, truncate=True, gpt2_type="gpt2")
val_dataset = Stories(validation_set, truncate=True, gpt2_type="gpt2")

# Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Accumulated batch size (since GPT2 is so big)


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

# --------------------------- 3. CREATE THE LOSS FUNCTION --------------------------------


def calculate_val_loss(validation_set, model):
    model.eval()
    device = device_name
    with torch.no_grad():
        val_dataloader = DataLoader(validation_set, batch_size=1)
        loss = 0
        input_tensor = None
        for idx, entry in tqdm(enumerate(val_dataloader)):
            entry = entry.to(device)
            (input_tensor, carry_on, remainder) = pack_tensor(
                entry, input_tensor, 768)
            if carry_on and idx != len(val_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
    return loss
# Where we had checkpoints in the collab for training:
# checkpoint_path =  "/content/gdrive/My Drive/CSC413/bestModel.pt"
# checkpoint2_latest =  "/content/gdrive/My Drive/CSC413/latestModel.pt"

# --------------------------- 4. TRAIN THE MODEL --------------------------------


def train(
    dataset, model, tokenizer,
    batch_size=4, epochs=2, lr=1e-6,
    max_seq_len=300, warmup_steps=250, output_dir=".",
    output_prefix="wreckgar", test_mode=False, save_model_on_epoch=False, validation_set=None
):
    acc_steps = 100
    device = device_name
    model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    iters = []
    train_losses = []
    val_losses = []
    i = 0
    best_val_loss = float('inf')
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        model.train()
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(
                entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            train_losses.append(loss.item())
            loss.backward()
            iters.append(i)
            i += 1

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None

        if validation_set:  # also recording initial validation loss for progress
            print("about to start validating")
            val_loss = calculate_val_loss(validation_set, model)
            val_losses.append(val_loss.item())

            if val_loss < best_val_loss:
                print("saving new model")
                best_val_loss = val_loss
            #   To save the checkpoints on google collab during training
            #   torch.save(model.state_dict(), checkpoint_path)

            # also save latest model, we might like some overfitting
            # To save the checkpoints on google collab during training
            # torch.save(model.state_dict(), checkpoint2_latest)

    return model, (iters, [i for i in range(epochs)], train_losses, val_losses)


model, plot_info = train(dataset, model, tokenizer,
                         epochs=10, validation_set=val_dataset)

# Get the model's training curve:


def plot_training_curve(iters, epochs, train_losses, valid_losses):
    # Training plot
    plt.title("Training Loss per Iter")
    plt.plot(iters, train_losses, label="training")
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.show()

    print("losses are", valid_losses, "epochs are", epochs)
    # Validation plot
    plt.title("Validation Loss per Epoch")
    plt.plot(epochs, valid_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


plot_training_curve(*plot_info)


# Load the best checkpoint
model2 = GPT2LMHeadModel.from_pretrained('gpt2', force_download=True)

# Loading the best checkpoint found during training (done on google collab)
# saved_state_dict = torch.load(checkpoint_path)
# model2.load_state_dict(saved_state_dict)


# --------------------------- 5. GENERATE THE STORY FROM THE PROMPT --------------------------------
def complete_prompt(prompt, model, min_length=50, max_length=200, top_p=0.8, temperature=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        do_sample=True,
        min_length=min_length,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    completed_story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return completed_story

# --------------------------- 6. TESTING THE MODEL --------------------------------
smoothie = SmoothingFunction().method4   # recommended in warning

model = model.cpu()

scores = []

for story in testing_set:
    if len(story.split(' ')) > 450:  # Can't evaluate long stories reliably
        continue

    characters_for_completion = 40
    prompt = story.split()[:-characters_for_completion]
    start = ' '.join(prompt)
    completion = story.split()[-characters_for_completion:]
    end = ' '.join(completion)

    candidate = complete_prompt(start, model, max_length=600)

    score = sentence_bleu(story, candidate, smoothing_function=smoothie)
    scores.append(score)

print(np.mean(scores))
print(complete_prompt("once in a while there was a ", model2))