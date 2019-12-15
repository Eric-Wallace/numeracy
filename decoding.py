from typing import Iterator, List, Dict
import os
import logging
import random
from overrides import overrides
import torch
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
import numpy as np
from num2words import num2words
import math
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import (TokenIndexer, SingleIdTokenIndexer)
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.iterators import BasicIterator
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, NoamLR
from allennlp.training.trainer import Trainer
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('number_decoding')

# a "dataset reader" whose "read" function just generates the data
class NumberDecodeGenerator(DatasetReader):
    def __init__(self,
                 max_interval: int = -1,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_word_form=False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.use_word_form = use_word_form
        self.max_interval = max_interval

    def text_to_instance(self, token: Token, target: int = None) -> Instance:
        number_text_field = TextField([token], self.token_indexers)
        fields = {"number_text": number_text_field}

        if target is not None:
            target_field = LabelField(target, skip_indexing=True)
            fields["target"] = target_field

        return Instance(fields)

    # we "center" the output to be mean 0. This makes it easier to learn, because the model will be
    # initialized to output 0, so if the target range is [0,200], the early learning time
    # will be spent just increasing the output to be magnitude 0. For extrapolation, I recommend
    # setting the range to be [-X, X], so you don't need centering.
    @overrides    
    def read(self, numbers: List = None, extrapolation = False) -> Iterator[Instance]:        
        for number in numbers:
            if self.use_word_form:
                word = num2words(number)
                if extrapolation:
                    yield self.text_to_instance(Token(word), number)
                else:
                    yield self.text_to_instance(Token(word), int(number - self.max_interval / 2))
            else:
                if extrapolation:
                    yield self.text_to_instance(Token(str(number)), int(number))
                else:
                    yield self.text_to_instance(Token(str(number)), int(number - self.max_interval / 2))


class NumberDecoder(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 hidden_dim: int,
                 linear: bool,
                 vocab: Vocabulary) -> None:            
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        if linear:
            self.decode_layer = torch.nn.Linear(in_features=self.word_embeddings.get_output_dim(),
                                                out_features=1)
        else:
            self.decode_layer = torch.nn.Sequential(torch.nn.Linear(in_features=word_embeddings.get_output_dim(),
                                                                    out_features=hidden_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_dim,
                                                                    out_features=hidden_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=hidden_dim,
                                                                    out_features=1))
        self.loss_func = MSELoss()

    def forward(self,
                number_text: Dict[str, torch.Tensor],
                target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embeddings = self.word_embeddings(number_text)
        prediction = self.decode_layer(embeddings)
        output = {"prediction": prediction, 'embeddings': embeddings}
        if target is not None:
            output["loss"] = self.loss_func(prediction.squeeze(-1).squeeze(-1), target.float())
        return output

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    
    train_numbers, val_numbers = get_intervals(args)


    data_reader = NumberDecodeGenerator(token_indexers=get_token_indexers(args.embedding),
                                        use_word_form=args.word_form,
                                        max_interval=args.max_interval)

    train_dataset = list(data_reader.read(train_numbers, extrapolation=args.extrapolation))
    validation_dataset = list(data_reader.read(val_numbers, extrapolation=args.extrapolation))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    if args.serialization_dir:
        vocab.save_to_files(os.path.join(args.serialization_dir, 'vocab'))

    model = NumberDecoder(word_embeddings=get_text_field_embedder(args.embedding, vocab, args.hidden_dim),
                          hidden_dim=args.hidden_dim,
                          vocab=vocab,
                          linear=args.linear)                          
    if args.cuda_device > -1:
        model.cuda(args.cuda_device)
    
    lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    iterator = BasicIterator(batch_size=20)
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      serialization_dir=args.serialization_dir,
                      patience=20,
                      grad_norm=5,
                      validation_metric='-loss',                      
                      num_epochs=args.num_epochs,
                      cuda_device=args.cuda_device)
    metrics = trainer.train()
    print(math.sqrt(metrics['best_validation_loss']))