import random
import numpy as np
import math
import argparse
from allennlp.data.token_indexers import (SingleIdTokenIndexer, TokenCharactersIndexer, 
                                          ELMoTokenCharactersIndexer, PretrainedBertIndexer)
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder, ElmoTokenEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import CnnEncoder, Seq2VecEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--embedding', type=str, default='random',
                        help='Choose from random, w2v, glove, char_cnn, char_lstm, elmo, bert')
    parser.add_argument('--word_form', action='store_true', default=False,
                        help='If true, uses word form, e.g., twenty-four')
    parser.add_argument('--linear', action='store_true', default=False,
                        help='Use a linear model. Only makes sense for decoding task')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help="the hidden dimension of the MLP layers or LSTM layers.")
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument('--cuda_device', type=int, default=0,
                        help="the gpu device to use.")
    parser.add_argument('-s', '--serialization_dir', type=str, default=None,
                        help='the directory to save training result.')

    # for interpolation mode (on by default) these set the train/test interval. This interval is shuffled and 80% is put into train and 20% into test.
    # for extrapolation, these set the training interval.    
    parser.add_argument('--min_interval', type=int, default=None,
                        help='Smallest number used, e.g., 0')
    parser.add_argument('--max_interval', type=int, default=None,
                        help='Biggest number used during training, e.g., 100')

    # turn on extrapolation mode
    parser.add_argument('--extrapolation', action='store_true', default=False,
                        help='If true, use extrapolation setting')
    # test interval for extrapolation, extrapola
    parser.add_argument('--min_extrapolation_test_interval', type=int, default=None,
                        help='Smallest number used during extrapolation testing, e.g., 0')
    parser.add_argument('--max_extrapolation_test_interval', type=int, default=None,
                        help='Biggest number used during extrapolation testing, e.g., 100')
    
    args = parser.parse_args()
    if args.extrapolation:
        assert args.min_extrapolation_test_interval is not None
        assert args.max_extrapolation_test_interval is not None
    else:
        assert args.min_extrapolation_test_interval is None
        assert args.max_extrapolation_test_interval is None
    return args

def get_text_field_embedder(embedding_type, vocab, hidden_dim=None):
    if embedding_type == "random":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), 
                                    embedding_dim=300,
                                    trainable=True)
        text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    
    elif embedding_type in ["glove", "w2v"]:
        pretrained_embedding_paths = {
            "glove": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "w2v": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",            
        }
        weight = _read_pretrained_embeddings_file(pretrained_embedding_paths[embedding_type],
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    elif embedding_type == "char_cnn" or embedding_type == "char_lstm":
        char_embedding_dim = 64
        if embedding_type == "char_cnn":
            char_encoder = CnnEncoder(char_embedding_dim, hidden_dim, ngram_filter_sizes=(2, 3, 4, 5))
        else:
            char_encoder = Seq2VecEncoder.by_name("lstm")(input_size=char_embedding_dim,
                                                          hidden_size=hidden_dim,
                                                          bidirectional=True)
        token_char_embedding = TokenCharactersEncoder(Embedding(vocab.get_vocab_size("token_characters"),
                                                                embedding_dim=char_embedding_dim),
                                                                char_encoder)
        text_field_embedder = BasicTextFieldEmbedder({"token_characters": token_char_embedding})

    elif embedding_type == "elmo": 
        elmo_embedding = ElmoTokenEmbedder(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                                           weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                                           do_layer_norm=False,
                                           dropout=0.0)
        text_field_embedder = BasicTextFieldEmbedder({"elmo": elmo_embedding})
    elif embedding_type == "bert":
        bert_embedding = PretrainedBertEmbedder(pretrained_model="bert-base-uncased",
                                                requires_grad=False,
                                                top_layer_only=True)
        text_field_embedder = BasicTextFieldEmbedder({"bert": bert_embedding},
                                                     embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
                                                     allow_unmatched_keys=True)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    return text_field_embedder

def get_token_indexers(embedding_type):
    if embedding_type in ["random", "w2v", "glove"]:
        indexer = {"tokens": SingleIdTokenIndexer()}
    elif embedding_type in ["char_cnn", "char_lstm"]:
        indexer = {"token_characters": TokenCharactersIndexer(min_padding_length=5)}
    elif embedding_type == "elmo":
        indexer = {"elmo": ELMoTokenCharactersIndexer()}
    elif embedding_type == "bert":
        indexer = {"bert": PretrainedBertIndexer(pretrained_model="bert-base-uncased",
                                                 do_lowercase=True,
                                                 use_starting_offsets=False)}
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    return indexer

def get_intervals(args):
    if args.extrapolation:  # extrapolation, train on train range and test on another test range
        train_numbers = list(range(args.min_interval, args.max_interval + 1))
        random.shuffle(train_numbers)  # just to make sure training set isn't in weird increasing order
        val_numbers = list(range(args.min_extrapolation_test_interval, args.max_extrapolation_test_interval + 1))

    else:  # interpolation, grab random 80% of range
        all_numbers = list(range(args.min_interval, args.max_interval + 1))
        random.shuffle(all_numbers)
        train_numbers = all_numbers[:int(0.8 * len(all_numbers))]
        val_numbers = all_numbers[int(0.8 * len(all_numbers)):]
    return  train_numbers, val_numbers
