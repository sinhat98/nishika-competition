CUDA_VISIBLE_DEVICES=0 poetry run python src/inference_parallel.py models --split_num 1 --config config/best_decode_config.yaml --write_header &
CUDA_VISIBLE_DEVICES=1 poetry run python src/inference_parallel.py models --split_num 2 --config config/best_decode_config.yaml &
CUDA_VISIBLE_DEVICES=2 poetry run python src/inference_parallel.py models --split_num 3 --config config/best_decode_config.yaml &
CUDA_VISIBLE_DEVICES=3 poetry run python src/inference_parallel.py models --split_num 4 --config config/best_decode_config.yaml &
