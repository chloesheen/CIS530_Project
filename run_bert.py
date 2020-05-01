"""
BERT model for author identification.
"""
import argparse
import math
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    XLNetConfig, 
    XLNetForSequenceClassification, 
    XLNetTokenizer,
    XLMConfig, 
    XLMForSequenceClassification, 
    XLMTokenizer,
    RobertaConfig, 
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    DistilBertConfig, 
    DistilBertForSequenceClassification, 
    DistilBertTokenizer,
    AlbertConfig, 
    AlbertForSequenceClassification, 
    AlbertTokenizer,
    CamembertConfig, 
    CamembertForSequenceClassification, 
    CamembertTokenizer,
    XLMRobertaConfig, 
    XLMRobertaForSequenceClassification, 
    XLMRobertaTokenizer,
    FlaubertConfig, 
    FlaubertForSequenceClassification, 
    FlaubertTokenizer,
    # ElectraConfig, 
    # ElectraForSequenceClassification, 
    # ElectraTokenizer,
    get_linear_schedule_with_warmup
)

from bert.callbacks import *
from bert.datasets import Reuters50Dataset
from bert.train import fit, predict_episode
from bert.utils import setup_dirs
from config import PATH, DATA_PATH

setup_dirs()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "camembert": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
    # "electra": (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--model-type', type=str, required=True)
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--max-length', type=int, default=512)
args = parser.parse_args()

batch_size = 8
epochs = 4
lr=4e-5
epsilon=1e-8

param_str = f'{args.model_type}_{args.model_name}_maxlen={args.max_length}'

###############
# Model Setup #
###############
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.model_name, num_labels=50)
model = model_class.from_pretrained(args.model_name, config=config)
tokenizer = tokenizer_class.from_pretrained(args.model_name,
                                            do_lower_case=False)

###################
# Create datasets #
###################
def get_encodings(texts):
    token_ids = []
    attention_masks = []
    for text in texts:
        token_id = tokenizer.encode(text, 
                                    add_special_tokens=True, 
                                    max_length=args.max_length,
                                    pad_to_max_length=True)
        token_ids.append(token_id)
    return token_ids


def get_attention_masks(padded_encodings):
    attention_masks = []
    for encoding in padded_encodings:
        attention_mask = [int(token_id > 0) for token_id in encoding]
        attention_masks.append(attention_mask)
    return attention_masks

train_df = pd.read_pickle(ROOT + '/reuters50_train.pkl')
test_df = pd.read_pickle(ROOT + '/reuters50_test.pkl')

train_encodings = get_encodings(train_df.text.values)
train_attention_masks = get_attention_masks(train_encodings)

test_encodings = get_encodings(test_df.text.values)
test_attention_masks = get_attention_masks(test_encodings)

train_input_ids = torch.tensor(train_encodings)
train_masks = torch.tensor(train_attention_masks)
train_labels = torch.tensor(train_df.author_id.values)


test_input_ids = torch.tensor(test_encodings)
test_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_df.author_id.values)


train_dataset = TensorDataset(train_input_ids, train_masks, train_labels)
# train_dataset = Reuters50Dataset('train', 
#                                  args.model_type, 
#                                  tokenizer,
#                                  args.max_length)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, 
                              sampler=train_sampler,
                              batch_size=batch_size)

test_dataset = TensorDataset(test_input_ids, test_masks, test_labels)
# test_dataset = Reuters50Dataset('test',
#                                 args.model_type,
#                                 tokenizer,
#                                 args.max_length)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=batch_size)


###################
# Optimizer Setup #
###################
optimizer = AdamW(model.parameters(), lr=lr, eps=epsilon)

total_steps = len(train_dataloader) * epochs
warmup_steps = math.ceil(total_steps * 0.06)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)


############
# Training #
############

callbacks = [
    EvaluateModel(
        eval_fn=predict_episode,
        dataloader=test_dataloader,
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/{param_str}.pth'
    ),
    CSVLogger(PATH + f'/logs/{param_str}.csv'),
]

fit(
    model,
    optimizer,
    scheduler,
    dataloader=train_dataloader,
    epochs=epochs,
    callbacks=callbacks,
)

