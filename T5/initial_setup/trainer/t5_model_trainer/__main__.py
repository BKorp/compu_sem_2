#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import sys
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from sklearn.model_selection import train_test_split
import zipfile

import random
import numpy as np


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('\nRunning the T5 model trainer!\n')



# In[2]:


with zipfile.ZipFile(sys.argv[0]) as zf:
    df = pd.read_csv(zf.open('sem-pmb_4_0_0-gold.csv'))


# In[3]:


df.head()


# In[4]:


grouped_sentences = df.groupby('sent_file').agg({'token': list, 'lemma': list, 'from': list, 'to': list, 'semtag': list}).reset_index()
del(df)


# In[5]:


print(grouped_sentences.head())


# In[6]:


output_sentences = []
for index, data in grouped_sentences.iterrows():
    current_sentence = []
    for word in data[1]:
        current_sentence.append('{}: '.format(data[-1][data[1].index(word)]))

        current_sentence.append(word)
        if word != data[1][-1]:
            current_sentence.append('; ')

    output_sentences.append(current_sentence)

grouped_sentences['output'] = output_sentences


# In[7]:


''.join(grouped_sentences['output'][2])


# In[8]:


' '.join(grouped_sentences.token[4])


# In[9]:


train_df = grouped_sentences.copy()

train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=SEED)

train_df.to_csv('train.csv')
test_df.to_csv('test.csv')
print('\nStored train and test!\n')

train_df.drop(['sent_file', 'lemma', 'semtag', 'from', 'to'], axis=1, inplace=True)
train_df.rename(columns={'token': 'input_text', 'output': 'target_text'}, inplace=True)
train_df.insert(0, 'prefix', 'semtag')


# In[10]:


train_df
train_df = train_df.iloc[  :35000,:]


# In[11]:


train_df

# In[12]:


'''Partially taken from https://github.com/MathewAlexander/T5_nlg - Edited by André Korporaal'''

def gpu_checker():
    '''Checking for the GPU availability'''
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    return dev


def model_trainer(train_df,
                  learning_rate=1e-3,
                  epochs=4,
                  batch_size=8,
                  padding=400,
                  model='t5-small',
                  extra_tokens=None):
    '''
    Trains given training data on a given model,
    default is t5-small
    '''
    print('Used parameters:\n')
    print(f'\nLearning rate: {learning_rate}\nEpochs: {epochs}\n'
          f'Batch_size: {batch_size}\nPadding: {padding}\n'
          f'Model: {model}\nExtra tokens: {extra_tokens}\n')

    train_df = train_df.sample(frac = 1)
    # batch_size = 8
    num_of_batches = round(len(train_df)/batch_size)
    print(f'\nNumber of batches: {num_of_batches}\n')
    num_of_epochs = epochs

    # Loading the pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model)
    if extra_tokens:
        new_tokens = tokenizer.additional_special_tokens + extra_tokens
        tokenizer = T5Tokenizer.from_pretrained(model, additional_special_tokens=new_tokens)
        print('\nSpecial tokens: ', tokenizer.get_added_vocab())

    model = T5ForConditionalGeneration.from_pretrained(model, return_dict=True)

    # moving the model to device(GPU/CPU)
    dev = gpu_checker()
    model.to(dev)

    # Initializing the Adafactor optimizer with parameter values suggested for t5
    optimizer = Adafactor(
        model.parameters(),
        lr = learning_rate,
        eps = (1e-30, 1e-3),
        clip_threshold = 1.0,
        decay_rate = -0.8,
        beta1 = None,
        weight_decay = 0.0,
        relative_step = False,
        scale_parameter = False,
        warmup_init = False
    )

    # num_of_epochs = 1

    # Training the model
    model.train()

    loss_per_10_steps = []
    for epoch in range(1, num_of_epochs + 1):
        print('\nRunning epoch: {}'.format(epoch))

        running_loss = 0

        for i in range(num_of_batches):
            inputbatch = []
            labelbatch = []
            new_df = train_df[i * batch_size:i * batch_size + batch_size]
            for indx,row in new_df.iterrows():
                train_prefix = row['prefix']
                train_text = str(' '.join(row['input_text']))
                target_text = str(' '.join(row['target_text']))
                # input = 'sem_tag: ' + row['input_text'] + '</s>'
                # input = f'{train_prefix}: ' + train_text + '</s>'
                input = f'{train_prefix}: ' + train_text
                # labels = target_text + '</s>'
                labels = target_text
                inputbatch.append(input)
                labelbatch.append(labels)
            inputbatch = tokenizer.batch_encode_plus(inputbatch, padding='max_length', max_length=padding, return_tensors='pt')["input_ids"]
            labelbatch = tokenizer.batch_encode_plus(labelbatch, padding='max_length', max_length=padding, return_tensors="pt")["input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            logits = outputs.logits
            running_loss += loss_num
            if i%10 == 0:
                loss_per_10_steps.append(loss_num)

            # calculating the gradients
            loss.backward()

            #updating the params
            optimizer.step()

        running_loss=running_loss/int(num_of_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))

    return model


def model_serializer(model, file_name):
    '''Serializes (and thus saves) the trained/finetuned model'''
    output_name = file_name[:-4]
    path_folder = './models'
    Path(path_folder).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(),f'{path_folder}/pytorch_model_{output_name}.bin')


# In[13]:


file_name = 'semtagging'

model = model_trainer(train_df, epochs=4, model='t5-base', extra_tokens=['~', 'ø'])
model_serializer(model, file_name)

