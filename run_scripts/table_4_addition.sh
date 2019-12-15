mkdir -p logs
mkdir -p logs/addition
export CUDA_VISIBLE_DEVICES=0;
# random
python addition.py --min_interval 0 --max_interval 99   --num_epochs 1 --embedding random > logs/addition/random_add_interp_0_99.log & 
python addition.py --min_interval 0 --max_interval 999  --num_epochs 1 --embedding random > logs/addition/random_add_interp_0_999.log &
python addition.py --min_interval 0 --max_interval 9999 --num_epochs 1 --embedding random > logs/addition/random_add_interp_0_9999.log  &

# glove
python addition.py --min_interval 0 --max_interval 99 --embedding glove > logs/addition/glove_add_interp_0_99.log  & 
python addition.py --min_interval 0 --max_interval 999 --embedding glove > logs/addition/glove_add_interp_0_999.log   &
python addition.py --min_interval 0 --max_interval 9999 --embedding glove > logs/addition/glove_add_interp_0_9999.log  &

# w2v
python addition.py --min_interval 0 --max_interval 99 --embedding w2v > logs/addition/w2v_add_interp_0_99.log &
python addition.py --min_interval 0 --max_interval 999 --embedding w2v > logs/addition/w2v_add_interp_0_999.log &
python addition.py --min_interval 0 --max_interval 9999 --embedding w2v > logs/addition/w2v_add_interp_0_9999.log &

# elmo
export CUDA_VISIBLE_DEVICES=1; python addition.py --min_interval 0 --max_interval 99  --embedding elmo > logs/addition/elmo_add_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=2; python addition.py --min_interval 0 --max_interval 999 --embedding elmo > logs/addition/elmo_add_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=3; python addition.py --min_interval 0 --max_interval 9999 --embedding elmo > logs/addition/elmo_add_interp_0_9999.log & 

# char_cnn
export CUDA_VISIBLE_DEVICES=4; python addition.py --min_interval 0 --max_interval 99 --embedding char_cnn > logs/addition/char_cnn_add_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=5; python addition.py --min_interval 0 --max_interval 999 --embedding char_cnn > logs/addition/char_cnn_add_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=6; python addition.py --min_interval 0 --max_interval 9999 --embedding char_cnn > logs/addition/char_cnn_add_interp_0_9999.log  &

# bert
export CUDA_VISIBLE_DEVICES=7; python addition.py --min_interval 0 --max_interval 99 --embedding bert > logs/addition/bert_add_interp_0_99.log
export CUDA_VISIBLE_DEVICES=0; python addition.py --min_interval 0 --max_interval 999 --embedding bert > logs/addition/bert_add_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=1; python addition.py --min_interval 0 --max_interval 9999 --embedding bert > logs/addition/bert_add_interp_0_9999.log & 

# char_lstm
export CUDA_VISIBLE_DEVICES=2; python addition.py --min_interval 0 --max_interval 99 --embedding char_lstm > logs/addition/char_lstm_add_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=3; python addition.py --min_interval 0 --max_interval 999 --embedding char_lstm > logs/addition/char_lstm_add_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=4; python addition.py --min_interval 0 --max_interval 9999 --embedding char_lstm > logs/addition/char_lstm_add_interp_0_9999.log &