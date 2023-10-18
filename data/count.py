import os
import json
def count_train_dev_test(task, dataset: str):
    splits = ['train.json', 'dev.json', 'test.json']
    paths = [os.path.join(task, dataset, s) for s in  splits]
    for p in paths:
        with open(p, 'r') as f:
            print(p)
            print(len(json.load(f)))
            
            
if __name__ == "__main__":
    t_d_dict = {'ner': ['zh_msra', 'resume', 'conll03'],
                'cws': ['msra', 'pku', 'ctb6'],
                'pos': ['ctb5']}
    
    for t, d in t_d_dict.items():
        for dd in d:
            count_train_dev_test(t, dd)