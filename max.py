from typing import Iterator, List, Dict, Any
import os
import torch
import logging
import random
import torch.optim as optim
from torch.nn.functional import nll_loss
from allennlp.data import Instance
from allennlp.data.fields import TextField, IndexField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import (TokenIndexer, SingleIdTokenIndexer)
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
import numpy as np
from num2words import num2words
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('number_max')

class NumberListGenerator(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 use_word_form=False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_word_form = use_word_form

    def text_to_instance(self, tokens: List[Token], target_index: int = None) -> Instance:
        number_text_field = TextField(tokens, self.token_indexers)
        fields = {"number_text": number_text_field}

        if target_index is not None:
            fields["target_index"] = IndexField(target_index, number_text_field)

        fields["metadata"] = MetadataField({'original_numbers': tokens})

        return Instance(fields)

    def _read(self, number_lists: List[List] = None) -> Iterator[Instance]:
        for number_list in number_lists:
            if self.use_word_form:
                yield self.text_to_instance([Token(num2words(number)) for number in number_list],
                                            number_list.index(max(number_list)))
            else:
                yield self.text_to_instance([Token(str(number)) for number in number_list],
                                            number_list.index(max(number_list)))

class NumberSelector(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 hidden_dim: int,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.decode_layer = Seq2SeqEncoder.by_name('lstm')(bidirectional=True,
                                                           input_size=self.word_embeddings.get_output_dim(),
                                                           hidden_size=hidden_dim,
                                                           num_layers=1)
        self.output_layer = torch.nn.Linear(self.decode_layer.get_output_dim(), 1)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                number_text: Dict[str, torch.Tensor],
                target_index: torch.Tensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        number_masks = get_text_field_mask(number_text).float()
        embeddings = self.word_embeddings(number_text)
        hidden_vectors = self.decode_layer(embeddings, number_masks)
        logits = self.output_layer(hidden_vectors).squeeze(-1)
        _, predictions = logits.max(-1)
        output = {"logits": logits}
        if target_index is not None:
            self.accuracy(logits, target_index.squeeze(-1))
            output["loss"] = nll_loss(masked_log_softmax(logits, number_masks), target_index.squeeze(-1))

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

# for the max task, there are some cases where the numbers are like {1,134,4532,938}
# and just looking at the length of the number will do well. So we try to cluster
# the numbers together to make the task harder. see the Appendix of the paper for more details.
# the logic below is properly more complicated than it needs to be
def generate_max_data(all_numbers, length, num_samples):
    def gaussian_sample(max_index):
        if max_index <= 500: # sample from a small Gaussian
            random_gaussian = np.random.normal(scale=5)
        else: # sample from a big Gaussian
            random_gaussian = np.random.normal(scale=max_index * 0.01)
        new_index = max_index - int(random_gaussian)
        if new_index >= max_index: # if out of bounds
            new_index = max_index - 1
        elif new_index < 0: # if out of bounds
            new_index = 0
        return new_index

    all_numbers = sorted(all_numbers)
    all_lists = []
    minimum = 0; maximum = len(all_numbers) - 1
    for i in range(num_samples):
        max_index = np.random.randint(low=10, high=maximum, size=1)[0] # sample a random number
        temp_list = [all_numbers[max_index]]
        if np.random.uniform() > 0.5: # for half the values, we just randomly sample
            for j in range(length - 1):
                new_int = all_numbers[np.random.randint(low=minimum, high=max_index-1, size=1)[0]]
                while new_int in temp_list: # resample if its already in there
                    new_int = all_numbers[np.random.randint(low=minimum, high=max_index-1, size=1)[0]]
                temp_list.append(new_int)
        else: # for the other half, we sample from a Gaussian to keep the numbers nearby
            for j in range(length - 1):                
                new_int = all_numbers[gaussian_sample(max_index)]
                while new_int in temp_list: # if already oresent, resample                    
                    new_int = all_numbers[gaussian_sample(max_index)]
                temp_list.append(new_int)

        random.shuffle(temp_list) # shuffle inside the examples
        all_lists.append(temp_list)
    random.shuffle(all_lists)
    return all_lists

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    # extrapolation, train on train range and test on another test range
    if args.extrapolation:
        all_numbers_train = list(range(args.min_interval, args.max_interval + 1))
        all_numbers_dev = list(range(args.min_extrapolation_test_interval, args.max_extrapolation_test_interval + 1))
        all_lists_train = generate_max_data(all_numbers_train, 5, 10000, False)
        all_lists_dev = generate_max_data(all_numbers_dev, 5, 1000, False)

    # interpolation, grab random 80% of range
    else:
        all_numbers = list(range(args.min_interval, args.max_interval + 1))
        random.shuffle(all_numbers)
        all_numbers_train = all_numbers[:int(0.80 * len(all_numbers))]
        all_numbers_dev = all_numbers[int(0.80 * len(all_numbers)):]
        all_lists_train = generate_max_data(all_numbers_train, 5, 100000)
        all_lists_dev = generate_max_data(all_numbers_dev, 5, 1000)

    data_reader = NumberListGenerator(token_indexers=get_token_indexers(args.embedding), use_word_form=args.word_form)

    train_dataset = list(data_reader.read(all_lists_train))
    validation_dataset = list(data_reader.read(all_lists_dev))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    if args.serialization_dir:
        vocab.save_to_files(os.path.join(args.serialization_dir, 'vocab'))

    model = NumberSelector(word_embeddings=get_text_field_embedder(args.embedding, vocab, args.hidden_dim),
                           hidden_dim=args.hidden_dim,
                           vocab=vocab)
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
                      patience=10,
                      validation_metric='+accuracy',
                      num_epochs=args.num_epochs,
                      cuda_device=args.cuda_device)
    metrics = trainer.train()
    print(metrics['best_validation_accuracy'])
    