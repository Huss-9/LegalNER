import json
import torch
from torch.utils.data import Dataset
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string 
import matplotlib.pyplot as plt
from collections import Counter


class NERDataset(Dataset):
    def __init__(self,
                 judgement_json, 
                 preamble_json,
                 entity_names,
                 label_to_index,
                 index_to_label):
        """
        Args:
            judgement_json (str): Path to judgement JSON file
            preamble_json (str): Path to preamble JSON file
            entity_names (list): List of entity names
            label_to_index (dict): Dictionary mapping entity names to indices
            index_to_label (dict): Dictionary mapping indices to entity names
        """

        with open(judgement_json, 'r') as f:
            self.data_judgement = json.load(f)

        with open(preamble_json, 'r') as f:
            self.data_preamble = json.load(f)

        self.data_list = self.data_judgement + self.data_preamble
        self.entity_names = entity_names
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label

    def __len__(self):
        return len(self.data_list)
    
    def get_entity_counts(self):
        entity_counts = Counter()

        for sample in self.data_list:
            annotations = sample['annotations'][0]['result']
            for annotation in annotations:
                label = annotation['value']['labels'][0]
                entity_counts[label] += 1
        
        return entity_counts

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        text = sample['data']['text']
        annotations = sample['annotations'][0]['result']

        # Preprocess text: remove punctuation symbols
        translator = str.maketrans('', '', string.punctuation)
        text_no_punct = text.translate(translator)

        words = word_tokenize(text_no_punct)
        labels_word_level = [0] * len(words)
        labels_char_level = [0] * len(text)
        entities = []

        # Function to find word indices based on character indices
        def find_word_index(char_index):
            subset = text[:char_index]
            subset = subset.translate(translator)
            return len(word_tokenize(subset))

        for annotation in annotations:
            start = annotation['value']['start']
            end = annotation['value']['end']
            label = annotation['value']['labels'][0]

            # Find the indices of tokens within the entity span
            token_start = find_word_index(start)
            token_end = find_word_index(end) 

            entities.append((token_start, token_end- 1, label))
            label_index = self.label_to_index.get(label, 0)

            labels_word_level[token_start:token_end] = [label_index] * (token_end - token_start)
            labels_char_level[start:end] = [label_index] * (end - start)

        return {
            'text': text,
            'words': words,
            'labels_word_level': torch.tensor(labels_word_level, dtype=torch.long),
            'labels_char_level': torch.tensor(labels_char_level, dtype=torch.long),
            'entities': entities
        }



# Example entity names and label mappings
entity_names = ["O", "COURT", "PETITIONER", "RESPONDENT", "JUDGE",
                "LAWYER", "DATE", "ORG", "GPE", "STATUTE", "PROVISION",
                "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"]
label_to_index = {label: idx for idx, label in enumerate(entity_names)}
index_to_label = {idx: label for idx, label in enumerate(entity_names)}

# Create NERDataset instances for train and development sets
train_dataset = NERDataset(judgement_json='Data/Train/NER_TRAIN_JUDGEMENT.json',
                         preamble_json='Data/Train/NER_TRAIN_PREAMBLE.json',
                         entity_names=entity_names,
                         label_to_index=label_to_index,
                         index_to_label=index_to_label)

dev_dataset = NERDataset(judgement_json='Data/Dev/NER_DEV_JUDGEMENT.json',
                         preamble_json='Data/Dev/NER_DEV_PREAMBLE.json',
                         entity_names=entity_names,
                         label_to_index=label_to_index,
                         index_to_label=index_to_label)


# EDA of the train and development datasets 
print("Lenght of the training dataset:", len(train_dataset))
sample = train_dataset[0]
print("Text:", sample['text'])
print("Words:", sample['words'])
print("Word-level Labels:", sample['labels_word_level'])
print("Char-level Labels:", sample['labels_char_level'])
print("Entities:", sample['entities'])
print() 

# Generation of Plots of the Train and Development datasets 
entity_counts_train = train_dataset.get_entity_counts()
entity_counts_dev = dev_dataset.get_entity_counts()

# Train 
sorted_entities = sorted(entity_counts_train.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*sorted_entities)
total_samples = len(train_dataset)
proportions = [count / total_samples for count in counts]

patches, texts, autotexts = plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
for text in texts + autotexts:
    text.set_fontsize(9)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Named Entity proportions in training.', pad=30, fontdict = {'fontsize':20, 'fontstyle' : 'oblique'})
plt.savefig('/media/axelrom16/HDD/AI/LegalNER/Plots/entity_proportions_train.png', bbox_inches='tight', transparent=True, dpi=160)

plt.clf()

# Dev 
sorted_entities = sorted(entity_counts_dev.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*sorted_entities)
total_samples = len(dev_dataset)
proportions = [count / total_samples for count in counts]

patches, texts, autotexts = plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
for text in texts + autotexts:
    text.set_fontsize(9)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Named Entity proportions in develpment.', pad=30, fontdict = {'fontsize':20, 'fontstyle' : 'oblique'})
plt.savefig('/media/axelrom16/HDD/AI/LegalNER/Plots/entity_proportions_dev.png', bbox_inches='tight', transparent=True, dpi=160)

