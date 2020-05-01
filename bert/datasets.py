from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from bert.utils import convert_example_to_features
from config import DATA_PATH

class Reuters50Dataset(Dataset):
    def __init__(self, subset, model_type, tokenizer, max_len=512):
        """Dataset class representing the Reuters 50 50 dataset.
        Reads entire dataset into memory to optimize for speed.

        # Arguments:
            subset: Whether the dataset represents the train or test set
        """
        if subset not in ('train', 'test'):
            raise(ValueError, 'subset must be one of (train, test)')
        self.subset = subset
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_len = max_len

        if subset == 'train':
            self.df = pd.read_pickle(DATA_PATH + "/reuters50_train.pkl")
        else:
            self.df = pd.read_pickle(DATA_PATH + "/reuters50_test.pkl")

    def num_classes(self):
        return len(self.df['author'].unique())

    def __getitem__(self, item):
        text = self.df.iloc[item].text
        label = self.df.iloc[item].author_id
        
        input_ids, input_mask = convert_example_to_features(text,
                                                            self.model_type,
                                                            self.tokenizer,
                                                            self.max_len)

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        label = torch.tensor(label)

        return input_ids, input_mask, label

    def __len__(self):
        return len(self.df)



# class Reuters50Dataset(Dataset):
#     def __init__(self, subset, model_type, tokenizer, max_len=512):
#         """Dataset class representing the Reuters 50 50 dataset.
#         Reads from disk to save memory.

#         # Arguments:
#             subset: Whether the dataset represents the train or test set
#         """
#         if subset not in ('train', 'test'):
#             raise(ValueError, 'subset must be one of (train, test)')
#         self.subset = subset
#         self.model_type = model_type
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#         # Add new column for id
#         self.df = pd.DataFrame(self.index_subset(self.subset))

#         # Convert authors to ordered indices
#         self.unique_authors = sorted(self.df['author'].unique())
#         self.author_to_id = {self.unique_authors[i]: i for i in
#                              range(self.num_classes())}
#         self.df = self.df.assign(author_id=self.df['author'].apply(lambda a:
#                                                                    self.author_to_id[a]))

#         # Create dicts
#         self.datasetid_to_filepath = self.df.to_dict()['filepath']
#         self.datasetid_to_author_id = self.df.to_dict()['author_id']

#     def num_classes(self):
#         return len(self.df['author'].unique())


#     def __getitem__(self, item):
#         text = ""
#         with open(self.datasetid_to_filepath[item], 'r') as f:
#             text = f.read()
        
#         input_ids, input_mask = convert_example_to_features(text,
#                                                             self.model_type,
#                                                             self.tokenizer,
#                                                             self.max_len)

#         label = self.datasetid_to_author_id[item]

#         input_ids = torch.tensor(input_ids)
#         input_mask = torch.tensor(input_mask)
#         label = torch.tensor(label)

#         return input_ids, input_mask, label

#     def __len__(self):
#         return len(self.df)

#     @staticmethod
#     def index_subset(subset):
#         """Index a subset by looping through all of its files and recording
#         relevant information.

#         # Arguments:
#             subset: Name of the subset

#         # Returns
#             A list of dicts containing information about all the text files in
#             a particular subset of the Reuters 50 50 dataset
#         """
#         texts = []
#         print('Indexing {}...'.format(subset))

#         # Quick first pass to find total for tqdm bar
#         subset_len = 0
#         for root, folders, files in os.walk(DATA_PATH +
#                                             '/C50/C50{}/'.format(subset)):
#             subset_len += len([f for f in files if f.endswith('.png')])

#         progress_bar = tqdm(total=subset_len)

#         for root, folders, files in os.walk(DATA_PATH +
#                                             '/C50/C50{}/'.format(subset)):
#             if len(files) == 0:
#                 continue

#             author = root.split('/')[-1]
            
#             for f in files:
#                 if f.endswith('.txt'):
#                     progress_bar.update(1)
#                     texts.append({
#                         'subset': subset,
#                         'author': author,
#                         'filepath': os.path.join(root, f)
#                     })

#         progress_bar.close()
#         return texts
