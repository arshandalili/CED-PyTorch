import json
import random
from pathlib import Path
import argparse
import numpy as np
from transformers import PreTrainedTokenizerFast


limit = 100
base_path = ''
save_path = ''




def tokenize_tweets(data_addr, data_class_addr, train_addr='', test_addr='', validation_adr='', eid_addr=''):

    tokenizer = PreTrainedTokenizerFast(tokenizer_file='twitter_vocab.json')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    with open(eid_addr) as file:
        eids = json.load(file)
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

    with open(train_addr, 'wt') as train_f, open(test_addr, 'wt') as test_f, open(validation_adr, 'wt') as valid_f:
        for eid, label in labels.items():
            eid_seq_ids = []
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
                        text_ids = tokenizer(text, padding='max_length', truncation=True, max_length=30)
                        eid_seq_ids.append(text_ids['input_ids'])
                else:
                    empty_tweets += 1
                    print(filename, 'does not exist in ', eid)

            real_len = len(engaged_users)

            if len(engaged_users) < limit and len(engaged_users) > 0:
                engaged_users.extend([random.choice(engaged_users) for i in range(limit - len(engaged_users))])
                eid_seq_ids.extend([random.choice(eid_seq_ids) for i in range(limit - len(eid_seq_ids))])

            engaged_users = engaged_users[0:limit]
            eid_seq_ids = eid_seq_ids[0:limit]

            if len(engaged_users) == 0:
                empty_eids += 1
            else:
                if label == 0:
                    real_events += 1
                else:
                    fake_events += 1

                json_data = json.dumps([{'seq': eid_seq_ids, 'len': real_len , 'eid' : eid}, labels[eid]])                

                if eid in eids['train']:
                    train_f.write(json_data + '\n')
                else:
                    if eid in eids['test']:
                        test_f.writelines(json_data + '\n')
                    elif eid in eids['validation']:
                        valid_f.writelines(json_data + '\n')
        

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)
parser.add_argument('--validation', type=str, required=True)
parser.add_argument('--limit', type=int, required=True)
args = parser.parse_args()
limit = args.limit


tokenize_tweets(base_path + 'twitter/', base_path + 'twitter/', save_path + args.train, save_path + args.test, save_path + args.validation, base_path + 'twitter/eids')

