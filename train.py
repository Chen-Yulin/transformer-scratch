import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # use code from https://huggingface.co/docs/tokenizers/quicktour
        special_tokens=['[UNK]', "[PAD]", "[SOS]", "[EOS]"]
        tokenizer = Tokenizer(WordLevel(lang, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer()
        tokenizer.train_from_iterator((get_all_sentenses(ds, lang), trainer=trainer))
