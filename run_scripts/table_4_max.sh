mkdir -p logs
mkdir -p logs/max
export CUDA_VISIBLE_DEVICES=0;
# random
python max.py --min_interval 0 --max_interval 99   --num_epochs 1 --embedding random > logs/max/random_max_interp_0_99.log & 
python max.py --min_interval 0 --max_interval 999  --num_epochs 1 --embedding random > logs/max/random_max_interp_0_999.log &
python max.py --min_interval 0 --max_interval 9999 --num_epochs 1 --embedding random > logs/max/random_max_interp_0_9999.log  &

# glove
python max.py --min_interval 0 --max_interval 99 --embedding glove > logs/max/glove_max_interp_0_99.log  & 
python max.py --min_interval 0 --max_interval 999 --embedding glove > logs/max/glove_max_interp_0_999.log   &
python max.py --min_interval 0 --max_interval 9999 --embedding glove > logs/max/glove_max_interp_0_9999.log  &

# w2v
python max.py --min_interval 0 --max_interval 99 --embedding w2v > logs/max/w2v_max_interp_0_99.log &
python max.py --min_interval 0 --max_interval 999 --embedding w2v > logs/max/w2v_max_interp_0_999.log &
python max.py --min_interval 0 --max_interval 9999 --embedding w2v > logs/max/w2v_max_interp_0_9999.log &

# elmo
export CUDA_VISIBLE_DEVICES=1; python max.py --min_interval 0 --max_interval 99  --embedding elmo > logs/max/elmo_max_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=2; python max.py --min_interval 0 --max_interval 999 --embedding elmo > logs/max/elmo_max_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=3; python max.py --min_interval 0 --max_interval 9999 --embedding elmo > logs/max/elmo_max_interp_0_9999.log & 

# char_cnn
export CUDA_VISIBLE_DEVICES=4; python max.py --min_interval 0 --max_interval 99 --embedding char_cnn > logs/max/char_cnn_max_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=5; python max.py --min_interval 0 --max_interval 999 --embedding char_cnn > logs/max/char_cnn_max_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=6; python max.py --min_interval 0 --max_interval 9999 --embedding char_cnn > logs/max/char_cnn_max_interp_0_9999.log  &

# bert
export CUDA_VISIBLE_DEVICES=7; python max.py --min_interval 0 --max_interval 99 --embedding bert > logs/max/bert_max_interp_0_99.log
export CUDA_VISIBLE_DEVICES=0; python max.py --min_interval 0 --max_interval 999 --embedding bert > logs/max/bert_max_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=1; python max.py --min_interval 0 --max_interval 9999 --embedding bert > logs/max/bert_max_interp_0_9999.log & 

# char_lstm
export CUDA_VISIBLE_DEVICES=2; python max.py --min_interval 0 --max_interval 99 --embedding char_lstm > logs/max/char_lstm_max_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=3; python max.py --min_interval 0 --max_interval 999 --embedding char_lstm > logs/max/char_lstm_max_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=4; python max.py --min_interval 0 --max_interval 9999 --embedding char_lstm > logs/max/char_lstm_max_interp_0_9999.log &