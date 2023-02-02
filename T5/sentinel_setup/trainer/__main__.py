import argparse
import numpy as np
import pandas as pd
import random
import sys
import torch
import zipfile
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers.optimization import Adafactor
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          MT5ForConditionalGeneration)


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parsing():
    '''Parses the expected command line arguments.'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default='t5-base', type=str,
                        help='Decide on which T5 model to use '
                             '(default: t5-base)')
    parser.add_argument('-lr', '--learning_rate', default=1e-3,
                        help='Set learning rate (default: 1e-3)')
    parser.add_argument('-ep', '--epochs', default=4, type=int,
                        help='Set epoch(s) (default: 4)')
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='Set batch size (default: 8)')
    parser.add_argument('-pd', '--padding', default=400, type=int,
                        help='Set the model padding (default: 400)')
    parser.add_argument('-es', '--early_stopping', default=None,
                        help='Set the amount of early stopping '
                             '(default: None)')
    parser.add_argument('-af', '--alternate_file', default=None,
                        help='Set alternative input csv or json '
                             '(default: None)')

    return parser.parse_args()


def data_prep(alternate_file=None):
    '''Extracts data or loads from existing files.'''
    output_path = f'./data'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    train_f = Path(f'{output_path}/train.json')
    dev_f = Path(f'{output_path}/dev.json')
    test_f = Path(f'{output_path}/test.json')

    if alternate_file or not train_f.is_file() or not dev_f.is_file() or \
       not test_f.is_file():

        if alternate_file:
            if alternate_file.split('.')[-1] == 'json':
                df = pd.read_json(alternate_file)
            else:
                df = pd.read_csv(alternate_file)

        else:
            with zipfile.ZipFile(sys.argv[0]) as zf:
                df = pd.read_csv(zf.open('sem-pmb_4_0_0-gold.csv'))

        grouped_sentences = df.groupby('sent_file').agg(
            {'token': list,
             'lemma': list,
             'from': list,
             'to': list,
             'semtag': list}).reset_index()

        print(grouped_sentences.head())

        in_token = []
        out_semtag = []
        for index, data in grouped_sentences.iterrows():
            cur_token = []
            cur_semtag = []
            cur_extra_id = -1
            for word in data[1]:
                cur_extra_id += 1
                cur_token.append('<extra_id_{}> {} '.format(
                    cur_extra_id,
                    word))
                cur_semtag.append('<extra_id_{}> {} '.format(
                    cur_extra_id,
                    data[-1][data[1].index(word)]))

            in_token.append(cur_token)
            out_semtag.append(cur_semtag)

        grouped_sentences['in_token'] = in_token
        grouped_sentences['out_semtag'] = out_semtag

        train_df = grouped_sentences.copy()

        train_df, devtest_df = train_test_split(
            train_df, test_size=0.2, random_state=SEED)
        dev_df, test_df = train_test_split(
            devtest_df, test_size=0.5, random_state=SEED)

        train_df.to_json(f'{output_path}/train.json')
        dev_df.to_json(f'{output_path}/dev.json')
        test_df.to_json(f'{output_path}/test.json')
        print('\nStored train and test!\n')

    else:
        train_df = pd.read_json(f'{output_path}/train.json')
        dev_df = pd.read_json(f'{output_path}/dev.json')

    train_df.drop(['sent_file', 'token', 'lemma', 'semtag',
                  'from', 'to'], axis=1, inplace=True)
    train_df.rename(columns={'in_token': 'input_text',
                    'out_semtag': 'target_text'}, inplace=True)
    train_df.insert(0, 'prefix', 'semtag')
    dev_df.drop(['sent_file', 'token', 'lemma', 'semtag',
                'from', 'to'], axis=1, inplace=True)
    dev_df.rename(columns={'in_token': 'input_text',
                  'out_semtag': 'target_text'}, inplace=True)
    dev_df.insert(0, 'prefix', 'semtag')

    return train_df, dev_df


def gpu_checker():
    '''Checks for GPU availability.'''
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    return dev


def train_one_epoch(model,
                    train_df,
                    num_of_batches,
                    batch_size,
                    tokenizer,
                    optimizer,
                    dev,
                    padding):
    '''Go through one training loop.'''
    running_loss = 0

    for i in range(num_of_batches):
        inputbatch = []
        labelbatch = []
        new_df = train_df[i * batch_size:i * batch_size + batch_size]
        for indx, row in new_df.iterrows():
            train_prefix = row['prefix']
            train_text = str(''.join(row['input_text']))
            target_text = str(''.join(row['target_text']))
            input = f'{train_prefix}: ' + train_text
            labels = target_text
            inputbatch.append(input)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(
            inputbatch,
            padding='max_length',
            max_length=padding,
            return_tensors='pt')["input_ids"]
        labelbatch = tokenizer.batch_encode_plus(
            labelbatch,
            padding='max_length',
            max_length=padding,
            return_tensors="pt")["input_ids"]

        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        loss = outputs.loss
        loss_num = loss.item()
        running_loss += loss_num

        # calculating the gradients
        loss.backward()

        # updating the params
        optimizer.step()

    running_loss = running_loss/int(num_of_batches)
    return model, optimizer, running_loss


def get_validation_loss(model,
                        dev_df,
                        num_of_batches,
                        batch_size,
                        tokenizer,
                        dev,
                        padding):
    '''Go through one validation loop.'''
    running_loss = 0

    for i in range(num_of_batches):
        inputbatch = []
        labelbatch = []
        new_df = dev_df[i * batch_size:i * batch_size + batch_size]
        for indx, row in new_df.iterrows():
            dev_prefix = row['prefix']
            dev_text = str(''.join(row['input_text']))
            target_text = str(''.join(row['target_text']))
            input = f'{dev_prefix}: ' + dev_text
            labels = target_text
            inputbatch.append(input)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(
            inputbatch,
            padding='max_length',
            max_length=padding,
            return_tensors='pt')["input_ids"]
        labelbatch = tokenizer.batch_encode_plus(
            labelbatch,
            padding='max_length',
            max_length=padding,
            return_tensors="pt")["input_ids"]
        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        loss = outputs.loss
        loss_num = loss.item()
        running_loss += loss_num

    running_loss = running_loss/int(num_of_batches)
    return running_loss


def model_trainer(train_df,
                  dev_df,
                  learning_rate=1e-3,
                  epochs=4,
                  batch_size=8,
                  padding=400,
                  model='t5-small',
                  earlyst_limit=None,
                  extra_tokens=None):
    '''Train a given model.

    Keyword arguments:
    learning_rate -- set the learning rate (default 1e-3)
    epochs -- set the maximum amount of epochs (default 4)
    batch_size -- set the batch size (default 8)
    padding -- set the max padding (default 400)
    model -- set the model (default t5-small)
    earlyst_limit -- set early stopping (default off)
    extra_tokens -- add special tokens to training (default off)
    '''

    print('Used parameters\n')
    print(f'Learning rate: {learning_rate}\nEpochs: {epochs}\n'
          f'Batch_size: {batch_size}\nPadding: {padding}\n'
          f'Model: {model}\nExtra tokens: {extra_tokens}\n'
          f'Early stopping: {earlyst_limit}\n')

    train_df = train_df.sample(frac=1)
    num_of_batches = round(len(train_df)/batch_size)
    num_of_dev_batches = round(len(dev_df)/batch_size)
    print(f'\nNumber of batches\ntrain: {num_of_batches}'
          f'\nvalidation (dev): {num_of_dev_batches}\n')

    num_of_epochs = epochs

    tokenizer = T5Tokenizer.from_pretrained(model)
    if extra_tokens:
        new_tokens = tokenizer.additional_special_tokens + extra_tokens
        tokenizer = T5Tokenizer.from_pretrained(
            model, additional_special_tokens=new_tokens)
        print('\nSpecial tokens: ', tokenizer.get_added_vocab())

    if 'mt5' in model.lower():
        model = MT5ForConditionalGeneration.from_pretrained(
            model, return_dict=True)
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model, return_dict=True)

    dev = gpu_checker()
    model.to(dev)

    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)

    best_vloss = 1_000_000
    earlyst_count = 0
    for epoch in range(1, int(num_of_epochs) + 1):

        model.train(True)

        model, optimizer, running_loss = train_one_epoch(model,
                                                         train_df,
                                                         num_of_batches,
                                                         batch_size,
                                                         tokenizer,
                                                         optimizer,
                                                         dev,
                                                         padding)

        model.train(False)

        # Still need to change the input to the validation set
        validation_loss = get_validation_loss(model,
                                              dev_df,
                                              num_of_dev_batches,
                                              batch_size,
                                              tokenizer,
                                              optimizer,
                                              dev,
                                              padding)

        print('Epoch {}/{}\nloss: {} - val_loss: {}'.format(epoch,
                                                            num_of_epochs,
                                                            running_loss,
                                                            validation_loss))

        if validation_loss < best_vloss:
            best_vloss = validation_loss
            best_model = model
            earlyst_count = 0

        elif earlyst_limit:
            if int(earlyst_count) > int(earlyst_limit):
                return best_model
            earlyst_count += 1

    return best_model


def model_serializer(model, file_name, model_name, timestring):
    '''Saves the trained/finetuned model'''
    output_name = file_name[:-4]
    path_folder = './models'
    Path(path_folder).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(),
               f'{path_folder}/{timestring}-pytorch_model'
               f'_{output_name}_{model_name}.bin')


def main():
    args = parsing()
    timestring = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    file_name = 'semtagging'
    train_df, dev_df = data_prep(args.alternate_file)

    model = model_trainer(train_df,
                          dev_df,
                          learning_rate=args.learning_rate,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          padding=args.padding,
                          model=args.model,
                          earlyst_limit=args.early_stopping,
                          extra_tokens=['~', 'ø'])

    original_model = args.model
    if '/' in original_model:
        original_model = original_model.split('/')[-1]
    model_serializer(model, file_name, original_model, timestring)


if __name__ == '__main__':
    main()
