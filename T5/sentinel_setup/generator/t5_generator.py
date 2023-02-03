import argparse
import numpy as np
import pandas as pd
import random
import re
import torch
from datetime import datetime
from pathlib import Path
from transformers import (T5Tokenizer,
                          MT5ForConditionalGeneration,
                          T5ForConditionalGeneration,
                          set_seed)


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)


class dataBuilder():
    '''Prepares the important parts of the test dataframe,
    such as gold and input labels.
    '''
    def __init__(self, test) -> None:
        self.test_df = pd.read_json(test)
        self._input_tokens()
        self._gold_tags()

    def gen_output_decoder(self, output_list, variant):
        '''Returns a list without special tokens.'''
        if variant == 'sentinel':
            split_token = '<extra_id_\\d*> *'

            return [re.split(split_token, re.sub(f'^{split_token}', '', i)) for
                    i in output_list]
        else:
            new_output_list = []
            for i in output_list:
                decode_list = []
                for i in re.split(' ; ', output_list):
                    if ':' in i:
                        decode_list.append(re.split(':', i)[0])
                    else:
                        decode_list.append('MISSING')
                new_output_list.append(decode_list)
            return new_output_list

    def store(self, lst, name, variant):
        '''Store a list into the the test data frame and export
        the dataframe into a json file with a given name.'''
        self.test_df['gen_semtag'] = self.gen_output_decoder(lst, variant)
        self.test_df.to_json(name)

    def _input_tokens(self):
        '''Store input tokens as a list into test_input.'''
        self.test_input = self.test_df.in_token.to_list()

    def _gold_tags(self):
        '''Store gold tags as a list into test_gold.'''
        self.test_gold = self.test_df.semtag.to_list()


class semtagGenerator():
    '''Prepares and Generates text-to-text output for semantic tagging.'''

    def __init__(self, model, config, original_model) -> None:
        self.model = model
        self.modelPrep(config, original_model)

    def modelPrep(self, config, original_model):
        '''Prepares the correct tokenizer and model for generation.'''
        extra_tokens = ['~', 'ø']

        if 'mt5' in original_model.lower():
            model = MT5ForConditionalGeneration.from_pretrained(
                self.model,
                return_dict=True,
                config=config
            )

        else:
            model = T5ForConditionalGeneration.from_pretrained(
                self.model,
                return_dict=True,
                config=config
            )

        self.dev = self.gpu_checker()
        model.to(self.dev)
        self.model = model

        tokenizer = T5Tokenizer.from_pretrained(
            original_model,
            model_max_length=512
        )
        new_tokens = tokenizer.additional_special_tokens + extra_tokens
        self.tokenizer = T5Tokenizer.from_pretrained(
            original_model,
            additional_special_tokens=new_tokens,
            model_max_length=512
        )

    def gpu_checker(self):
        '''Checking for the GPU availability'''
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            dev = torch.device("cpu")
            print("Running on the CPU")

        return dev

    def generate(self, input_list, prefix):
        '''Returns a list of tagged sentences (generated).'''
        output_list = []
        for sentence in input_list:

            input_ids = self.tokenizer.encode(
                "{}: {}".format(prefix, sentence),
                return_tensors="pt"
            ).to(self.dev)

            outputs = self.model.generate(input_ids,
                                          num_beams=10,
                                          max_length=500).to(self.dev)
            gen_text = self.tokenizer.decode(outputs[0])
            gen_text = gen_text.replace('<pad>', '').replace('</s>', '')
            gen_text = gen_text.lstrip().rstrip()

            output_list.append(gen_text)

        return output_list


def parsing():
    '''Parses the expected command line arguments.'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',
                        default='t5-base',
                        type=str,
                        help='Decide on which T5 model '
                        'to use (default: t5-base)')
    parser.add_argument('-om', '--original_model',
                        default='t5-base',
                        type=str,
                        help='Decide on which T5 model '
                        'to use (default: t5-base)')
    parser.add_argument('-mc', '--model_config', default='', type=str,
                        help='Give the config that you should download '
                        'from huggingface.co/<model>/blob/main/config.json')
    parser.add_argument('-ts', '--test_data', default='', type=str,
                        help='Use the test data that was created during '
                        'the training')
    parser.add_argument('-v', '--variant', default=None,
                        help='Choose between Sentinel and '
                        'Original (None) encoding (default: None)')

    return parser.parse_args()


def main():
    args = parsing()

    now = datetime.now()
    timestring = now.strftime('%Y-%m-%d_%H-%M-%S')
    output_path = 'generated'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    model = args.model
    modelconfig = args.model_config
    original_model = args.original_model

    data = dataBuilder(args.test_data)

    t5_gen = semtagGenerator(model, modelconfig, original_model)
    data.store(
        t5_gen.generate(
            data.test_input,
            prefix='semtag'),
        f'{output_path}/{timestring}-{original_model}-test_generated.json',
        variant=args.variant)


if __name__ == '__main__':
    main()
