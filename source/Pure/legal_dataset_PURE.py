"""
Script to transform the original JSON data into a JSON with the specific structure for the PURE Model. 
"""
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import json
import string 
import random
random.seed(42)


# Function to find word indices based on character indices
def find_word_index(text, char_index):
    subset = text[:char_index]
    #subset = subset.translate(translator)
    return len(word_tokenize(subset))

# Function to convert the structure of the JSON data
def convert_json_structure(input_json):
    doc_key = input_json["id"]
    sentence = input_json["data"]["text"]

    # Preprocess text: remove punctuation symbols and tokenize
    #translator = str.maketrans('', '', string.punctuation)
    #text_no_punct = sentence.translate(translator)
    tokens = word_tokenize(sentence)

    ner = []

    for annotation in input_json["annotations"][0]["result"]:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        label = annotation["value"]["labels"][0]

        # Find the indices of tokens within the entity span
        token_start = find_word_index(sentence, start)
        token_end = find_word_index(sentence, end) 

        ner.append([token_start, token_end-1, label])
    
    if len(ner) == 0:
        ner.append([])

    output_json = {
        "doc_key": doc_key,
        "sentences": [tokens], #[sentence.split() for sentence in sentences],
        "ner": [ner]
    }

    return output_json


# Import JSON files (train and dev)
with open("Data/Train/NER_TRAIN_JUDGEMENT.json", 'r') as f:
    judgement_train = json.load(f)
with open("Data/Train/NER_TRAIN_PREAMBLE.json", 'r') as f:
    preamble_train = json.load(f)
train_data = judgement_train + preamble_train

with open("Data/Dev/NER_DEV_JUDGEMENT.json", 'r') as f:
    judgement_dev = json.load(f)
with open("Data/Dev/NER_DEV_PREAMBLE.json", 'r') as f:
    preamble_dev = json.load(f)
dev_data = judgement_dev + preamble_dev

# Generate test set randomly from train set
random.shuffle(judgement_train)
judgement_test = judgement_train[:400]
judgement_train = judgement_train[400:]

random.shuffle(preamble_train)
preamble_test = preamble_train[:100]
preamble_train = preamble_train[100:]

test_data = judgement_test + preamble_test
train_data = judgement_train + preamble_train

random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(dev_data)

print("Train data length: ", len(train_data))
print("Test data length: ", len(test_data))
print("Dev data length: ", len(dev_data))

# Convert the structure of the 3 sets of data and store the new JSON files
train_data_converted = [convert_json_structure(case) for case in train_data]
test_data_converted = [convert_json_structure(case) for case in test_data]
dev_data_converted = [convert_json_structure(case) for case in dev_data]

with open("Data/PURE_data/train.json", "w") as output_file:
    for train in train_data_converted:
        if train["ner"] != [[[]]]:
            json.dump(train, output_file)
            output_file.write("\n")
        else:
            continue
    #json.dump(train_data_converted[0], output_file, indent=2)

with open("Data/PURE_data/test.json", "w") as output_file:
    for test in test_data_converted:
        if test["ner"] != [[[]]]:
            json.dump(test, output_file)
            output_file.write("\n")
        else:
            continue
    #json.dump(test_data_converted[0], output_file, indent=2)

with open("Data/PURE_data/dev.json", "w") as output_file:
    for dev in dev_data_converted:
        if dev["ner"] != [[[]]]:
            json.dump(dev, output_file)
            output_file.write("\n")
        else:
            continue
    #json.dump(dev_data_converted[0], output_file, indent=2)

