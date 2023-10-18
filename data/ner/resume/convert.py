import json

def transform(path: str, out_path: str):
    with open(path, 'r') as f:
        data = f.read()
    data = data.split('\n\n')
    new_data = []
    print(f"data length: {len(data)}")
    
    for d in data:
        if not d:
            continue
        sentence, label = [], []
        for bigram in d.split('\n'):
            try:
                s, l = bigram.split()
            except:
                print(d)
                print(bigram)
                import pdb;
                pdb.set_trace()
            sentence.append(s)
            label.append(l)

        i = 0
        new_label = []
        while i < len(label):
            if label[i] == 'O':
                new_label.append('O')
                i += 1
            elif label[i].startswith("B-"):
                ent = label[i][2:]
                start = i
                i += 1
                while i < len(label) and label[i] == 'I-' + ent:
                    i += 1
                end = i
                if (end - start) == 1:
                    new_label.append('S-' + ent)
                else:
                    new_label.append('B-' + ent)
                    new_label.extend(['M-' + ent] * (end - start - 2))
                    new_label.append('E-' + ent)
            else:
                new_label.append("O")
                i += 1
        try:
            assert len(label) == len(new_label) == len(sentence)
        except:
            import pdb;
            pdb.set_trace()
        new_data.append({'sentence': sentence, 
                         'label': new_label})
    
    
    with open(out_path, 'w') as f:
        json.dump(new_data, f, indent=1, ensure_ascii=False)
        
        
        
if __name__ == "__main__":
    transform("test.txt", 'test.json')
    