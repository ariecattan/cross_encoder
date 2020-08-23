import torch
from torch.utils import data
import collections
from itertools import combinations
import numpy as np
import os
import json


class CrossEncoderDataset(data.Dataset):
    def __init__(self, mentions_repr, first, second, labels):
        self.instances = [' '.join([mentions_repr[first[i]], "[SEP]",
                                    mentions_repr[second[i]], "[SEP]"]) for i in range(len(first))]
        self.labels = labels.to(torch.float)


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)




class CrossEncoderDatasetInstances(data.Dataset):
    def __init__(self, instances):
        self.instances = instances


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index]



class CrossEncoderDatasetFull(data.Dataset):
    def __init__(self, config, split_name, same_lemma=False):
        # self.corpus = corpus
        # self.documents = corpus.documents
        # self.mentions = corpus.mentions

        self.read_files(config, split_name)
        self.lemmas = np.asarray([x['lemmas'] for x in self.mentions])
        self.topics = set([m['topic'] for m in self.mentions])
        self.mention_labels = torch.tensor([m['cluster_id'] for m in self.mentions])
        self.doc_dict = self.make_dict_of_sentences(self.documents)


        self.mentions_by_topics = collections.defaultdict(list)
        for i, m in enumerate(self.mentions):
            self.mentions_by_topics[m['topic']].append(i)

        self.first = []
        self.second = []
        self.labels = []

        for topic, mentions in self.mentions_by_topics.items():
            first, second = zip(*list(combinations(range(len(mentions)), 2)))
            mentions = torch.tensor(mentions)
            first, second = torch.tensor(first), torch.tensor(second)
            first, second = mentions[first], mentions[second]
            labels = (self.mention_labels[first] != 0) & (self.mention_labels[second] != 0) \
                     & (self.mention_labels[first] == self.mention_labels[second])

            self.first.extend(first)
            self.second.extend(second)
            self.labels.extend(labels)

        self.first = torch.tensor(self.first)
        self.second = torch.tensor(self.second)
        self.labels = torch.tensor(self.labels, dtype=torch.float)


        if same_lemma:
            idx = (self.lemmas[self.first] == self.lemmas[self.second]).nonzero()
            self.first = self.first[idx]
            self.second = self.second[idx]
            self.labels = self.labels[idx]



        self.instances = self.prepare_pair_of_mentions(self.mentions, self.first,
                                                       self.second)



    def read_files(self, config, split_name):
        docs_path = os.path.join(config.data_folder, split_name + '.json')
        mentions_path = os.path.join(config.data_folder,
                                     split_name + '_{}.json'.format(config.mention_type))
        with open(docs_path, 'r') as f:
            self.documents = json.load(f)

        self.mentions = []
        if config.use_gold_mentions:
            with open(mentions_path, 'r') as f:
                self.mentions = json.load(f)



    def make_dict_of_sentences(self, documents):
        doc_dict = {}
        for doc, tokens in documents.items():
            dict = collections.defaultdict(list)
            for i, (sentence_id, token_id, text, flag) in enumerate(tokens):
                dict[sentence_id].append([token_id, sentence_id, text, flag])
            doc_dict[doc] = dict

        return doc_dict


    def encode_mention_with_context(self, mention):
        doc_id, sentence_id = mention['doc_id'], int(mention['sentence_id'])
        tokens = self.doc_dict[doc_id][sentence_id]
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_ids']))
        end_idx = token_ids.index(max(mention['tokens_ids'])) + 1

        mention_repr = [x[2] for x in tokens[:start_idx]] + ["[START]"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["[END]"] \
                       + [x[2] for x in tokens[end_idx:]]

        return ' '.join(mention_repr)


    def prepare_mention_representation(self, mentions):
        return np.asarray([self.encode_mention_with_context(m) for m in mentions])


    def prepare_pair_of_mentions(self, mentions, first, second):
        mentions_repr = np.asarray([self.encode_mention_with_context(m) for m in mentions])
        instances = [' '.join([mentions_repr[first[i]], "[SEP]",
                                    mentions_repr[second[i]], "[SEP]"]) for i in range(len(first))]

        return instances



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)





class CrossEncoderDatasetTopic(data.Dataset):
    def __init__(self, full_dataset, topic):
        super(CrossEncoderDatasetTopic, self).__init__()
        self.topic_mentions_ids = full_dataset.mentions_by_topics[topic]
        self.topic_mentions = [full_dataset.mentions[x] for x
                               in self.topic_mentions_ids]

        first, second = zip(*list(combinations(range(len(self.topic_mentions)), 2)))
        self.first, self.second = torch.tensor(first), torch.tensor(second)

        self.instances = full_dataset.prepare_pair_of_mentions(
            self.topic_mentions, self.first, self.second)



    def __len__(self):
        return len(self.topic_mentions)


    def __getitem__(self, index):
        return self.instances[index]

