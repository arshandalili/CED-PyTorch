import json
from pathlib import Path
import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, Unigram
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


base_path = ''
save_path = ''


def prepare_vocabulary(data_addr, data_class_addr, dataset):
    dataset_text = []
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(vocab_size=20000)
    tokenizer.pre_tokenizer = Whitespace()
    labels = {}
    seqs = {}
    with open(data_class_addr) as f:
        for line in f.readlines():
            eid = str(line[line.index('eid:') + 4: line.index('label')-1])
            label = int(line[line.index('label:') + 6])
            seq = line[line.index('label:') + 8 : line.index('\n')]
            seq = seq.split()
            seqs[eid] = seq
            labels[eid] = label
            all_tweets += len(seq)
    for eid, label in labels.items():
        engaged_users = []
        tweet_ind = 0
        while tweet_ind < len(seqs[eid]):
            tweet = seqs[eid][tweet_ind]
            tweet_ind += 1
            filename = tweet + '.txt'
            addr = data_addr + '/' + eid + '-' + str(label)
            if Path(addr + '/' + filename).exists():
                with open(addr + '/' + filename, encoding="utf8") as f:
                    event_data = json.load(f)
                    user = event_data['tweet']
                    engaged_users.append(user)
                    text = event_data['tweet']['text'].lower()
                    dataset_text.append(text)
            else:
                empty_tweets += 1
                print(filename, 'not exist in ', eid)
    tokenizer.train_from_iterator(dataset_text, trainer)
    tokenizer.save(f"{dataset}_vocab.json")

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)
parser.add_argument('--validation', type=str, required=True)
parser.add_argument('--limit', type=int, required=True)
args = parser.parse_args()
limit = args.limit


prepare_vocabulary(base_path + 'twitter', base_path + 'twitter', 'twitter')
