mkdir -p logs/decoding
mkdir -p logs/decoding


export CUDA_VISIBLE_DEVICES=0;
# random
python decoding.py --min_interval 0 --max_interval 99   --num_epochs 1 --embedding random > logs/decoding/random_decode_interp_0_99.log & 
python decoding.py --min_interval 0 --max_interval 999  --num_epochs 1 --embedding random > logs/decoding/random_decode_interp_0_999.log &
python decoding.py --min_interval 0 --max_interval 9999 --num_epochs 1 --embedding random > logs/decoding/random_decode_interp_0_9999.log  &

# glove
python decoding.py --min_interval 0 --max_interval 99 --embedding glove > logs/decoding/glove_decode_interp_0_99.log  & 
python decoding.py --min_interval 0 --max_interval 999 --embedding glove > logs/decoding/glove_decode_interp_0_999.log   &
python decoding.py --min_interval 0 --max_interval 9999 --embedding glove > logs/decoding/glove_decode_interp_0_9999.log  &

# w2v
python decoding.py --min_interval 0 --max_interval 99 --embedding w2v > logs/decoding/w2v_decode_interp_0_99.log &
python decoding.py --min_interval 0 --max_interval 999 --embedding w2v > logs/decoding/w2v_decode_interp_0_999.log &
python decoding.py --min_interval 0 --max_interval 9999 --embedding w2v > logs/decoding/w2v_decode_interp_0_9999.log &

# elmo
export CUDA_VISIBLE_DEVICES=1; python decoding.py --min_interval 0 --max_interval 99  --embedding elmo > logs/decoding/elmo_decode_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=2; python decoding.py --min_interval 0 --max_interval 999 --embedding elmo > logs/decoding/elmo_decode_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=3; python decoding.py --min_interval 0 --max_interval 9999 --embedding elmo > logs/decoding/elmo_decode_interp_0_9999.log & 

# char_cnn
export CUDA_VISIBLE_DEVICES=4; python decoding.py --min_interval 0 --max_interval 99 --embedding char_cnn > logs/decoding/char_cnn_decode_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=5; python decoding.py --min_interval 0 --max_interval 999 --embedding char_cnn > logs/decoding/char_cnn_decode_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=6; python decoding.py --min_interval 0 --max_interval 9999 --embedding char_cnn > logs/decoding/char_cnn_decode_interp_0_9999.log  &

# bert
export CUDA_VISIBLE_DEVICES=7; python decoding.py --min_interval 0 --max_interval 99 --embedding bert > logs/decoding/bert_decode_interp_0_99.log
export CUDA_VISIBLE_DEVICES=0; python decoding.py --min_interval 0 --max_interval 999 --embedding bert > logs/decoding/bert_decode_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=1; python decoding.py --min_interval 0 --max_interval 9999 --embedding bert > logs/decoding/bert_decode_interp_0_9999.log & 

# char_lstm
export CUDA_VISIBLE_DEVICES=2; python decoding.py --min_interval 0 --max_interval 99 --embedding char_lstm > logs/decoding/char_lstm_decode_interp_0_99.log &
export CUDA_VISIBLE_DEVICES=3; python decoding.py --min_interval 0 --max_interval 999 --embedding char_lstm > logs/decoding/char_lstm_decode_interp_0_999.log &
export CUDA_VISIBLE_DEVICES=4; python decoding.py --min_interval 0 --max_interval 9999 --embedding char_lstm > logs/decoding/char_lstm_decode_interp_0_9999.log &