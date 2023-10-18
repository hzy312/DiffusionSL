"""
@Date  : 2023/1/31
@Time  : 21:24
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import os
import json
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizer


class LabelSet:
    def __init__(self, dataset):
        postfix = ['train.json', 'dev.json', 'test.json']
        paths = [os.path.join(os.getcwd(), 'data', 'cws', dataset, p) for p in postfix]
        self._labelset = set()
        for p in paths:
            with open(p, 'r') as f:
                data = json.load(f)
            for d in data:
                self._labelset.update(set(d['label']))
        self._labelset = sorted(list(self._labelset), reverse=True)
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, 'S')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)


class CWSDataset(Dataset):
    def __init__(self, dataset: str, mode: str, label_set: LabelSet):
        super(CWSDataset, self).__init__()
        path = os.path.join(os.getcwd(), 'data', 'cws', dataset, mode + '.json')
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.label_set = label_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, label = self.data[item]['sentence'], self.data[item]['label']
        label = [self.label_set.label2id(l) for l in label]
        return sentence, label


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sentences, labels = map(list, zip(*batch))
        inputs_encoding = self.tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True,
                                         max_length=self.max_length)
        input_ids = inputs_encoding.input_ids
        attention_mask = inputs_encoding.attention_mask
        word_ids = [inputs_encoding.word_ids(i) for i in range(len(batch))]

        seq_labels = []
        for ids, las in zip(word_ids, labels):
            temp = [i if i is not None else -100 for i in ids]
            seql = []
            for i in range(len(temp)):
                if temp[i] == -100:
                    seql.append(-100)
                else:
                    if temp[i] != temp[i - 1]:
                        seql.append(las[temp[i]])
                    else:
                        seql.append(-100)
            seq_labels.append(seql)

        assert len(seq_labels) == len(sentences)
        assert len(seq_labels[0]) == len(input_ids[0])

        return torch.as_tensor(input_ids, dtype=torch.long), \
               torch.as_tensor(attention_mask, dtype=torch.long), \
               torch.as_tensor(seq_labels, dtype=torch.long)


class Collator_from_scratch:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _pad_to_right(self, inputs, target_length, padding_value):
        outputs = [i + ([padding_value] * (target_length - len(i))) for i in inputs]
        return outputs

    def __call__(self, batch):
        sentences, labels = map(list, zip(*batch))
        input_ids = []
        attention_mask = []
        seq_labels = []
        for s, l in zip(sentences, labels):
            ii = []
            sl = []
            for i, c in enumerate(s):
                pieces = self.tokenizer.tokenize(c)
                ii.extend(self.tokenizer.convert_tokens_to_ids(pieces))
                sl.append(l[i])
                sl.extend([-100] * (len(pieces) - 1))
            # truncation to max_length if surpass the limit
            if len(ii) > self.max_length:
                ii = ii[:self.max_length]
                sl = sl[:self.max_length]
            am = [1] * len(ii)
            assert len(ii) == len(sl) == len(am)
            input_ids.append(ii)
            attention_mask.append(am)
            seq_labels.append(sl)

        max_length_in_batch = max([len(ii) for ii in input_ids])
        input_ids = self._pad_to_right(input_ids, max_length_in_batch, self.tokenizer.pad_token_id)
        attention_mask = self._pad_to_right(attention_mask, max_length_in_batch, 0)
        seq_labels = self._pad_to_right(seq_labels, max_length_in_batch, -100)

        return torch.as_tensor(input_ids, dtype=torch.long), \
               torch.as_tensor(attention_mask, dtype=torch.long), \
               torch.as_tensor(seq_labels, dtype=torch.long)
