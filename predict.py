from conll import write_output_file
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder, FullCrossEncoder
from utils import *

import argparse
import pyhocon
from dataset import CrossEncoderDatasetFull, CrossEncoderDatasetTopic, CrossEncoderDatasetInstances
from sklearn.cluster import AgglomerativeClustering
from itertools import product
import collections
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import RobertaModel




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering_cross_encoder.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    data = CrossEncoderDatasetFull(config, config.split)

    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    cross_encoder = FullCrossEncoder(config).to(device)
    cross_encoder.model = RobertaModel.from_pretrained(os.path.join(config.model_path, 'bert')).to(device)
    cross_encoder.linear.load_state_dict(torch.load(os.path.join(config.model_path, 'linear')))
    model = torch.nn.DataParallel(cross_encoder, device_ids=device_ids)
    model.eval()

    clustering_5 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.5)
    clustering_55 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.55)
    clustering_6 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.6)
    clustering_65 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.65)
    clustering_7 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.7)

    clustering = [clustering_5, clustering_55, clustering_6, clustering_65, clustering_7]
    clusters = [list(), list(), list(), list(), list()]
    max_ids = [0, 0, 0, 0, 0]
    threshold = {id: thresh for id, thresh in enumerate([0.5, 0.55, 0.6, 0.65, 0.7])}



    doc_ids, sentence_ids, starts, ends = [], [], [], []
    logger.info('Number of topics: {}'.format(len(data.topic_list)))


    for topic_num, topic in enumerate(data.topics):
        logger.info('Processing topic {}'.format(topic))
        topic_dataset = CrossEncoderDatasetTopic(data, topic)
        topic_loader = torch.utils.data.DataLoader(topic_dataset, batch_size=64)

        all_scores = []
        topic_mentions_ids = data.mentions_by_topics[topic]
        topic_mentions = [data.mentions[x] for x in topic_mentions_ids]

        mentions_repr = data.prepare_mention_representation(topic_mentions)

        doc_ids.extend([m['doc_id'] for m in topic_mentions])
        sentence_ids.extend([m['sentence_id'] for m in topic_mentions])
        starts.extend([min(m['tokens_ids']) for m in topic_mentions])
        ends.extend([max(m['tokens_ids']) for m in topic_mentions])


        first, second = zip(*list(product(range(len(topic_mentions)), repeat=2)))
        first, second = torch.tensor(first), torch.tensor(second)

        instances = [' '.join([mentions_repr[first[i]], "[SEP]",
                               mentions_repr[second[i]], "[SEP]"]) for i in range(len(first))]

        for i in tqdm(range(0, len(instances), 512)):
            batch = [instances[x] for x in range(i, min(i + 512, len(instances)))]
            bert_tokens = model.module.tokenizer(batch, pad_to_max_length=True)
            input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

            with torch.no_grad():
                scores = model(input_ids, attention_mask)
                scores = torch.sigmoid(scores)
                all_scores.extend(scores.detach().cpu().squeeze(1))



        all_scores = torch.stack(all_scores)

        # Affinity score to distance score
        pairwise_distances = 1 - all_scores.view(len(mentions_repr), len(mentions_repr)).numpy()



        for i, agglomerative in enumerate(clustering):
            predicted = agglomerative.fit(pairwise_distances)
            predicted_clusters = predicted.labels_ + max_ids[i]
            max_ids[i] = max(predicted_clusters) + 1
            clusters[i].extend(predicted_clusters)


    for i, predicted in enumerate(clusters):
        logger.info('Saving cluster for threshold {}'.format(threshold[i]))
        all_clusters = collections.defaultdict(list)
        for span_id, cluster_id in enumerate(predicted):
            all_clusters[cluster_id].append(span_id)

        # removing singletons
        all_clusters = {cluster_id:mentions for cluster_id, mentions in all_clusters.items()
                           if len(mentions) > 1}

        print('Saving conll file...')
        doc_name = '{}_{}_{}_{}'.format(
            config['split'], config['mention_type'], config['linkage_type'], threshold[i])

        write_output_file(data.documents, all_clusters, doc_ids, starts, ends, config['save_path'], doc_name,
                          topic_level=config.topic_level, corpus_level=not config.topic_level)
