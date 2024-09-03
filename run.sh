CUDA_VISIBLE_DEVICES=0 poetry run python src/inference_parallel.py models --split_num 1 --config config/decode_config_v10.yaml --write_header &
CUDA_VISIBLE_DEVICES=1 poetry run python src/inference_parallel.py models --split_num 2 --config config/decode_config_v10.yaml &
CUDA_VISIBLE_DEVICES=2 poetry run python src/inference_parallel.py models --split_num 3 --config config/decode_config_v10.yaml &
CUDA_VISIBLE_DEVICES=3 poetry run python src/inference_parallel.py models --split_num 4 --config config/decode_config_v10.yaml &
