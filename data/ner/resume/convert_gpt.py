import json

def transform(path, output_path):
    with open(path, 'r') as f:
        data = f.read()
        
    data = data.split("\n\n")
    new_data = []
    all_categories = set()
    for d in data:
        sentence, labels = [], []
        for bi in d.split("\n"):
            try:
                word, label = bi.split(" ")
            except:
                import pdb;
                pdb.set_trace()
            sentence.append(word)
            labels.append(label)
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                start = i
                entity_category = labels[i][2:]
                all_categories.add(entity_category)
                i += 1
                while i < len(labels) and labels[i] == 'I-' + entity_category:
                    i += 1
                end = i
                entities.append({'text': ''.join(sentence[start: end]),
                                 'label': entity_category})
            else:
                i += 1
        sentence = ''.join(sentence)
        new_data.append({'sentence': sentence,
                         'entities': entities})
    with open(output_path, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        
    print(all_categories)
            
            
if __name__ == '__main__':
    transform('train.txt', 'train_gpt.json')