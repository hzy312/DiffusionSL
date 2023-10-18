import os.path as osp
import datasets
from datasets import Features, Value

logger = datasets.logging.get_logger(__name__)

_URL = 'http://mitalinlp.oss-cn-hangzhou.aliyuncs.com/adaseq/datasets/resume.zip'

PAD_LABEL = 'X'
PAD_LABEL_ID = -100


class ResumeNERConfig(datasets.BuilderConfig):
    '''BuilderConfig for ResumeNER'''

    def __init__(self, data_url, delimiter=None, **kwargs):
        '''BuilderConfig for ResumeNER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        '''
        super(ResumeNERConfig, self).__init__(version=datasets.Version('1.0.0'), **kwargs)
        self.data_url = data_url
        self.delimiter = delimiter


class ResumeNER(datasets.GeneratorBasedBuilder):
    '''ResumeNER dataset.'''

    BUILDER_CONFIGS = [ResumeNERConfig(name='default', data_url=_URL)]
    DEFAULT_CONFIG_NAME = 'default'

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features(
                {
                    'id': Value('string'),
                    'tokens': [Value('string')],
                    'spans': [
                        {
                            'start': Value('int32'),  # close
                            'end': Value('int32'),  # open
                            'type': Value('string'),
                        }
                    ],
                    'mask': [Value('bool')],
                }
            )
        )
        return info

    def _split_generators(self, dl_manager):
        '''Returns SplitGenerators.'''
        data_dir = dl_manager.download_and_extract(self.config.data_url)
        
        data_files = dict()
        for split_name in ['train', 'dev', 'test']:
            data_file = osp.join(data_dir, f'{split_name}.txt')
            if osp.exists(data_file):
                data_files[split_name] = data_file

        return [
            datasets.SplitGenerator(
                name=split_name, gen_kwargs={'filepath': data_files[split_name]}
            )
            for split_name in data_files.keys()
        ]

    def _generate_examples(self, filepath):
        return self._load_column_data_file(filepath, delimiter=self.config.delimiter)

    @classmethod
    def _load_column_data_file(cls, filepath, delimiter):
        with open(filepath, encoding='utf-8') as f:
            guid = 0
            tokens = []
            labels = []
            for line in f:
                if line.startswith('# id') or line == '' or line == '\n':
                    if tokens:
                        spans = cls._labels_to_spans(labels)
                        mask = cls._labels_to_mask(labels)
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'spans': spans,
                            'mask': mask,
                        }
                        guid += 1
                        tokens = []
                        labels = []
                else:
                    splits = line.split(delimiter)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans, 'mask': mask}

    @classmethod
    def _labels_to_spans(cls, labels):
        spans = []
        in_entity = False
        start = -1
        for i in range(len(labels)):
            # fix label error
            if labels[i][0] in 'IE' and not in_entity:
                labels[i] = 'B' + labels[i][1:]
            if labels[i][0] in 'BS':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    start = i
                else:
                    spans.append({'start': i, 'end': i + 1, 'type': labels[i][2:]})
            elif labels[i][0] in 'IE':
                if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                    assert start >= 0, 'Invalid label sequence found: {}'.format(labels)
                    spans.append({'start': start, 'end': i + 1, 'type': labels[i][2:]})
                    start = -1
            if labels[i][0] in 'B':
                in_entity = True
            elif labels[i][0] in 'OES':
                in_entity = False
        return spans

    @classmethod
    def _labels_to_mask(cls, labels):
        mask = []
        for label in labels:
            mask.append(label != PAD_LABEL)
        return mask
