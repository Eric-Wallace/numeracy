# Do NLP Models Know Numbers? Probing Numeracy in Embeddings

This is the official code for the 2019 EMNLP paper, [Do NLP Models Know Numbers? Probing Numeracy in Embeddings
](https://arxiv.org/abs/1909.07940). This repository contains the code for the synthetic tasks described in Section 3 of the paper.

## Installation

This code is written in python using PyTorch and AllenNLP.

An easy way to install the code is to create a fresh anaconda environment:

```
conda create -n numeracy python=3.6
source activate numeracy
pip install -r requirements.txt
```
Now you should be ready to go!

## Files

The three tasks in the paper (list maximum, number decoding, and addition) are separated into three different files. `utils.py` contains code for loading the embedding methods and creating the datasets.

## Example Runs

Examples here.

## References

Please consider citing our paper if you found this code or our work beneficial to your research.
```
@inproceedings{Wallace2019Numbers,
                Author = {Eric Wallace and Yizhong Wang and Sujian Li and Sameer Singh and Matt Gardner},
                Booktitle = {Empirical Methods in Natural Language Processing},                            
                Year = {2019},
                Title = {Do NLP Models Know Numbers? Probing Numeracy in Embeddings}}
```

## Contact

For issues with code or suggested improvements, feel free to open a pull request.

To contact the authors, reach out to Eric Wallace (ericwallace@berkeley.edu) and Yizhong Wang (yizhongw@cs.washington.edu
).