
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import PreTrainedTokenizer


class LabelSet1D:
    def __init__(self, dataset):
        postfix = ['train.json', 'dev.json', 'test.json']
        paths = [os.path.join(os.getcwd(), 'data', 'ner', dataset, p) for p in postfix]
        self._labelset = set()
        for p in paths:
            with open(p, 'r') as f:
                data = json.load(f)
            for d in data:
                self._labelset.update(set(d['label']))
        self._labelset = sorted(list(self._labelset), key=lambda x: (x[2:], x[0]))
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, 'O')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)


class LabelSet2D:
    def __init__(self, dataset):
        postfix = ['train2d.json', 'dev2d.json', 'test2d.json']
        paths = [os.path.join(os.getcwd(), 'data', dataset, p) for p in postfix]
        self._labelset = set()
        for p in paths:
            with open(p, 'r') as f:
                data = json.load(f)
            for d in data:
                self._labelset.update(set([n['type'] for n in d['ner']]))
        self._labelset_ = ['PAD', 'SUC']  # padding succession
        self._labelset_.extend(list(self._labelset))
        self._labelset = self._labelset_
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, 'PAD')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)


class LabelSetGP:
    def __init__(self, dataset):
        postfix = ['train.json', 'dev.json', 'test.json']
        paths = [os.path.join(os.getcwd(), 'data', dataset, p) for p in postfix]
        self._labelset = set()
        self._max_length = 0
        for p in paths:
            with open(p, 'r') as f:
                data = json.load(f)
            for d in data:
                self._labelset.update(set([n['type'] for n in d['ner']]))
        self._labelset = sorted(list(self._labelset))
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, 'OTHER')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)


class LabelSetNEW:
    def __init__(self, dataset):
        postfix = ['train.json', 'dev.json', 'test.json']
        paths = [os.path.join(os.getcwd(), 'data', dataset, p) for p in postfix]
        self._labelset = set()
        self._max_length = 0
        for p in paths:
            with open(p, 'r') as f:
                data = json.load(f)
            for d in data:
                self._labelset.update(set([n['type'] for n in d['ner']]))
        self._labelset = ['OTHER'] + sorted(list(self._labelset))
        self._label2id = {label: i for i, label in enumerate(self._labelset)}
        self._id2label = {i: label for i, label in enumerate(self._labelset)}
        assert self.id2label(self.label2id('OTHER')) == 'OTHER'
        assert self.label2id(self.id2label(0)) == 0

    def label2id(self, label: str):
        return self._label2id[label]

    def id2label(self, idx: int):
        return self._id2label.get(idx, 'OTHER')

    def __str__(self):
        string = [f"{v}:\t{k}" for k, v in self._label2id.items()]
        return '\n'.join(string)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._labelset)


class NERDataset1D(Dataset):
    def __init__(self, dataset: str, mode: str, label_set: LabelSet1D):
        super(NERDataset1D, self).__init__()
        path = os.path.join(os.getcwd(), 'data', 'ner', dataset, mode + '.json')
        with open(path, 'r') as f:
            self.data = json.load(f)
            if mode == 'train':
                import random
                self.data = random.sample(self.data, 20)
        self.label_set = label_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, label = self.data[item]['sentence'], self.data[item]['label']
        label = [self.label_set.label2id(l) for l in label]
        return sentence, label


class NERDataset2D(Dataset):
    def __init__(self, dataset: str, mode: str):
        super(NERDataset2D, self).__init__()
        path = os.path.join(os.getcwd(), 'data', dataset, mode + '2d.json')
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, ner = self.data[item]['sentence'], self.data[item]['ner']
        return sentence, ner


class NERDatasetGP(Dataset):
    def __init__(self, dataset: str, mode: str):
        super(NERDatasetGP, self).__init__()
        path = os.path.join(os.getcwd(), 'data', dataset, mode + '.json')
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, ner = self.data[item]['sentence'], self.data[item]['ner']
        return sentence, ner

class NERDatasetNEW(Dataset):
    def __init__(self, dataset: str, mode: str):
        super(NERDatasetNEW, self).__init__()
        path = os.path.join(os.getcwd(), 'data', dataset, mode + '.json')
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence, ner = self.data[item]['sentence'], self.data[item]['ner']
        return sentence, ner


class Collator1D:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
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


class Collator2D:
    def __init__(self, tokenizer: PreTrainedTokenizer, labelset: LabelSet2D, max_length: int = 256):
        self.tokenizer = tokenizer
        self.labelset = labelset
        self.max_length = max_length

    def __call__(self, batch, mode='train'):
        assert mode in ['train', 'dev', 'test']
        sentences, ners = map(list, zip(*batch))
        sentences = [s[:self.max_length] for s in sentences]
        if mode == 'train':
            ners = [[e for e in ner if e['index'][-1] < self.max_length] for ner in ners]

        inputs_encoding = self.tokenizer(sentences, padding=True, truncation=True,
                                         max_length=self.max_length)
        input_ids = inputs_encoding.input_ids
        attention_mask = inputs_encoding.attention_mask

        word_ids = [inputs_encoding.word_ids(i) for i in range(len(batch))]
        word_lens = [len(s) for s in sentences]
        max_word_len = max(word_lens)
        word2pieces = torch.zeros(size=[len(sentences), max_word_len, len(input_ids[0])])
        for i in range(len(sentences)):
            for j in range(word_lens[i]):
                word2pieces[i, j] = torch.LongTensor([1 if k == j else 0 for k in word_ids[i]])
        word2pieces = word2pieces.bool()

        grid_masks = torch.zeros(size=[len(sentences), max_word_len, max_word_len])
        for i in range(len(sentences)):
            grid_masks[i, :word_lens[i], :word_lens[i]] = 1
        grid_masks = grid_masks.bool()
        if mode != 'train':
            return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
                   word2pieces, None, grid_masks, ners, word_lens

        grid_labels = torch.zeros(size=[len(sentences), max_word_len, max_word_len], dtype=torch.long)
        for i in range(len(sentences)):
            for entity in ners[i]:
                index = entity['index']
                for j in range(len(index) - 1):
                    grid_labels[i, index[j], index[j + 1]] = 1
                grid_labels[i, index[-1], index[0]] = self.labelset.label2id(entity['type'])

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
               word2pieces, grid_labels, grid_masks, ners, word_lens


class CollatorGP:
    def __init__(self, tokenizer: PreTrainedTokenizer, labelset: LabelSetGP, max_length: int = 256):
        self.tokenizer = tokenizer
        self.labelset = labelset
        self.max_length = max_length

    def __call__(self, batch, mode='train'):
        assert mode in ['train', 'dev', 'test']
        sentences, ners = map(list, zip(*batch))

        inputs_encoding = self.tokenizer(sentences, is_split_into_words=True, padding='max_length', truncation=True,
                                         max_length=self.max_length, return_tensors='pt')
        input_ids = inputs_encoding.input_ids
        bsz, seq_len = input_ids.shape
        attention_mask = inputs_encoding.attention_mask

        # [bsz, seq_len]
        word_ids = [[j if j is not None else -100 for j in inputs_encoding.word_ids(i)] for i in range(bsz)]

        grid_labels = torch.zeros(size=[bsz, seq_len, seq_len, len(self.labelset)], dtype=torch.long)
        for i in range(bsz):
            for entity in ners[i]:
                start, end = entity['index'][0], entity['index'][-1]
                # index the first piece of word
                try:
                    s, e = word_ids[i].index(start), word_ids[i].index(end)
                except:
                    continue
                grid_labels[i, s, e, self.labelset.label2id(entity['type'])] = 1
        word_ids = torch.LongTensor(word_ids)
        assert word_ids.shape == input_ids.shape
        valid_seq_lens = attention_mask.sum(dim=-1)

        grid_masks = torch.zeros(size=[bsz, seq_len, seq_len])

        for i in range(bsz):
            grid_masks[i, :valid_seq_lens[i], :valid_seq_lens[i]] = 1
        grid_masks = grid_masks.bool()
        return input_ids, attention_mask, grid_labels, grid_masks


class CollatorNEW:
    def __init__(self, tokenizer: PreTrainedTokenizer, labelset: LabelSetNEW, max_length: int = 256):
        self.tokenizer = tokenizer
        self.labelset = labelset
        self.max_length = max_length

    def __call__(self, batch, mode='train'):
        assert mode in ['train', 'dev', 'test']
        sentences, ners = map(list, zip(*batch))

        inputs_encoding = self.tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True,
                                         max_length=self.max_length, return_tensors='pt')
        input_ids = inputs_encoding.input_ids
        bsz, seq_len = input_ids.shape
        attention_mask = inputs_encoding.attention_mask

        # [bsz, seq_len]
        word_ids = [[j if j is not None else -100 for j in inputs_encoding.word_ids(i)] for i in range(bsz)]

        grid_labels = torch.zeros(size=[bsz, seq_len, seq_len], dtype=torch.long)
        for i in range(bsz):
            for entity in ners[i]:
                start, end = entity['index'][0], entity['index'][-1]
                # index the first piece of word
                try:
                    s, e = word_ids[i].index(start), word_ids[i].index(end)
                except:
                    continue
                grid_labels[i, s, e] = self.labelset.label2id(entity['type'])
        word_ids = torch.LongTensor(word_ids)
        assert word_ids.shape == input_ids.shape
        valid_seq_lens = attention_mask.sum(dim=-1)

        grid_masks = torch.zeros(size=[bsz, seq_len, seq_len])

        for i in range(bsz):
            grid_masks[i, :valid_seq_lens[i], :valid_seq_lens[i]] = 1
        grid_masks = grid_masks.bool()
        return input_ids, attention_mask, grid_labels, grid_masks