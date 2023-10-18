import json
from tqdm import tqdm

def calculate_f1(path):
    with open(path, 'r') as f:
        data = json.load(f)
        
    
    num_tp, num_pred, num_gold = 0, 0, 0
    times = 0
    for d in tqdm(data):
        predicted = d['predicted']
        golds = d['entities']
        num_gold += len(golds)

        try:
            a1, a2, a3, a4, a5, a6, a7, a8 = predicted.split('；')
        except:
            times += 1
            continue
        a1 = [] if a1.endswith('无') else a1.split('：')[-1].split('，')
        a2 = [] if a2.endswith('无') else a2.split('：')[-1].split('，')
        a3 = [] if a3.endswith('无') else a3.split('：')[-1].split('，')
        a4 = [] if a4.endswith('无') else a4.split('：')[-1].split('，')
        a5 = [] if a5.endswith('无') else a5.split('：')[-1].split('，')
        a6 = [] if a6.endswith('无') else a6.split('：')[-1].split('，')
        a7 = [] if a7.endswith('无') else a7.split('：')[-1].split('，')
        a8 = [] if a8.endswith('无') else a8.split('：')[-1].split('，')
        preds = []
        for e in a1:
            preds.append({'text': e, 'label': 'CONT'})
        for e in a2:
            preds.append({'text': e, 'label': 'EDU'})
        for e in a3:
            preds.append({'text': e, 'label': 'LOC'})
        for e in a4:
            preds.append({'text': e, 'label': 'NAME'})
        for e in a5:
            preds.append({'text': e, 'label': 'ORG'})
        for e in a6:
            preds.append({'text': e, 'label': 'PRO'})
        for e in a7:
            preds.append({'text': e, 'label': 'RACE'})
        for e in a8:
            preds.append({'text': e, 'label': 'TITLE'})


        for g in golds:
            try:
                if g in preds:
                    num_tp += 1
            except:
                break
        num_pred += len(preds) if preds is not None else 0
        
        
    p = num_tp / num_pred
    r = num_tp / num_gold
    print(f'num_tp: {num_tp}')
    print(f'num_pred: {num_pred}')
    print(f'num_gold: {num_gold}')
    f = 2 * p * r / (p + r)
    print(f"p: {p}, r: {r}, f: {f}")
    
    print(times)
        
        
if __name__ == "__main__":
    calculate_f1('res_15shot.json')