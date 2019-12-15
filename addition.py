import math
from typing import Iterator, List, Dict
import torch
import logging
import random
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
import numpy as np
from num2words import num2words

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.iterators import BasicIterator
from allennlp.training.trainer import Trainer
import random
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('number_addition')

class NumberAdditionGenerator(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 max_interval: int = -1,
                 use_word_form=False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers
        self.max_interval = max_interval
        self.use_word_form = use_word_form

    def text_to_instance(self, tokens: List[Token], target: int = None) -> Instance:
        number_text_field = TextField(tokens, self.token_indexers)
        fields = {"number_text": number_text_field}

        if target is not None:
            target_filed = LabelField(target, skip_indexing=True)
            fields["target"] = target_filed

        return Instance(fields)

    def _read(self, tuples: List = None, extrapolation = False) -> Iterator[Instance]:
        for number_1, number_2 in tuples:            
            if self.use_word_form:                
                if extrapolation:
                    yield self.text_to_instance([Token(num2words(number_1)), Token(num2words(number_2))], int(number_1 + number_2))
                else:                    
                    yield self.text_to_instance([Token(num2words(number_1)), Token(num2words(number_2))], int(number_1 + number_2) - int((self.max_interval + self.max_interval - 1) / 2))
            else:
                if extrapolation:
                    yield self.text_to_instance([Token(str(number_1)), Token(str(number_2))], int(number_1 + number_2))
                else:
                    yield self.text_to_instance([Token(str(number_1)), Token(str(number_2))], int(number_1 + number_2) - int((self.max_interval + self.max_interval - 1) / 2))


class NumberAddition(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 hidden_dim: int,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.decode_layer = torch.nn.Sequential(torch.nn.Linear(in_features=word_embeddings.get_output_dim() * 2,
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
        concated_number_embeddings = torch.cat([embeddings[:, 0, :], embeddings[:, 1, :]], dim=-1)
        prediction = self.decode_layer(concated_number_embeddings)
        output = {"prediction": prediction}
        if target is not None:
            output["loss"] = self.loss_func(prediction.squeeze(-1).squeeze(-1), target.float())
        return output

# random pairs of examples from the data
def generate_addition_data(dataset, size):
    all_tuples = []
    for i in range(size):
        first_number = random.choice(dataset)
        second_number = random.choice(dataset)
        while second_number == first_number:
            second_number = random.choice(dataset)
        all_tuples.append((first_number,second_number))

    random.shuffle(all_tuples)
    return all_tuples

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    
    train_numbers, val_numbers = get_intervals(args)
 
    all_tuples_for_train = generate_addition_data(train_numbers, 100000)
    all_tuples_for_dev = generate_addition_data(val_numbers, 1000)
    
    data_reader = NumberAdditionGenerator(token_indexers=get_token_indexers(args.embedding),
                                          use_word_form=args.word_form,
                                          max_interval=args.max_interval)

    train_dataset = list(data_reader.read(all_tuples_for_train))
    validation_dataset = list(data_reader.read(all_tuples_for_dev))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    model = NumberAddition(word_embeddings=get_text_field_embedder(args.embedding, vocab, args.hidden_dim),
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
                      patience=20,
                      grad_norm=5,
                      validation_metric='-loss',
                      num_epochs=args.num_epochs,
                      cuda_device=args.cuda_device)
    metrics = trainer.train()
    print(math.sqrt(metrics['best_validation_loss']))
