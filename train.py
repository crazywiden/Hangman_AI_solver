import os
import time
import logging
import torch
import torch.nn as nn
from model import Word2Batch, RNN_model
CUDA = torch.cuda.is_available()


def show_game(original_word, guesses, obscured_words_seen):
    print('Hidden word was "{}"'.format(original_word))
    for i in range(len(guesses)):
        word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
        print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))


def get_all_words(file_location):
    with open(file_location, "r") as text_file:
        all_words = text_file.read().splitlines()
    return all_words


# get data
root_path = os.getcwd()
file_name = "words_250000_train.txt"
file_path = os.path.join(root_path, file_name)
words = get_all_words(file_path)
num_words = len(words)

# define model
model = RNN_model(target_dim=26, hidden_units=16)

# define hyper parameter
n_epoch = 2
lr = 0.001
record_step = 100  # output result every 100 words
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
loss_func = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)

# start training
start_time = time.perf_counter()
tot_sample = 0
for n in range(n_epoch):
    i = 0

    while tot_sample < (n + 1) * num_words:
        word = words[i]
        if len(word) == 1:
            continue
        i += 1
        # generate data in a batch
        new_batch = Word2Batch(word=word, model=model)
        obscured_word, prev_guess, correct_response = new_batch.game_mimic()
        if CUDA:
            obscured_word = obscured_word.cuda()
        optimizer.zero_grad()
        predict = model(obscured_word, prev_guess)
        loss = loss_func(predict, correct_response)
        loss.backward()
        optimizer.step()
        # show loss
        curr_time = time.perf_counter()
        print("for word {}, the BCE loss is {:4f}, time used:{:4f}".format(word, loss.item(), curr_time - start_time))
        # show guess status
        if i % record_step == 0:
            guesses = [chr(i+97) for i in torch.argmax(prev_guess, 1)]
            show_game(word, guesses, obscured_word)
        tot_sample += 1



