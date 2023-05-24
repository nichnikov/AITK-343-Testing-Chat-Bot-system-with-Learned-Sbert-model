import os
# import fasttext
from src.my_parser import (parser, 
                           tokenizer)
from src.config import (
    parameters,
    logger,
    PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier
from transformers import (BertTokenizer,
                          BertModelWithHeads,
                          )
import torch

# ft_model = fasttext.load_model(os.path.join(PROJECT_ROOT_DIR, "models", "bss_cbow_lem_clear.bin"))

# dct = Dictionary.load(os.path.join(PROJECT_ROOT_DIR, "dicts", "GENERAL4.gensim"))
# lsi_model = LsiModel.load(os.path.join(PROJECT_ROOT_DIR, "models", "GENERAL.lsi100_4.gensim"))

# classifier = FastAnswerClassifier(dct, lsi_model, parser, parameters, ft_model)
model_name = "bert-base-multilingual-cased"
transformer_tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)

adapter_name = "nli_adapter"
mode_name = "checkpoint-190000"
# mode_name = "nli-adapter-bert-big"
adapter_path = os.path.join(os.getcwd(), "models", mode_name)
adapter_path2 = os.path.join(os.getcwd(), "models", mode_name, adapter_name)

model.load_adapter(adapter_path2)
model.set_active_adapters(adapter_name)

# classifier = FastAnswerClassifier(parser, parameters, model, transformer_tokenizer)
classifier = FastAnswerClassifier(tokenizer, parameters, model, transformer_tokenizer)
logger.info("service started...")
