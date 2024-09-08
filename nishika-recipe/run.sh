#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=valid
test_sets="test"

asr_config=conf/train_asr_reazon_ft.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm_reazon.yaml

pretrained_model="reazonspeech-espnet-v2/exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_10best.pth"

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"


# Assuming Stage1 creating `data`, so you can skip it if you have `data`.
./asr.sh \
 --ngpu 4 \
 --lang jp \
 --token_type char \
 --feats_type raw \
 --asr_config "${asr_config}" \
 --inference_config "${inference_config}" \
 --lm_config "${lm_config}" \
 --train_set "${train_set}" \
 --valid_set "${valid_set}" \
 --test_sets "${test_sets}" \
 --lm_train_text "data/train/text" \
 --lm_dev_text "data/valid/text" \
 --speed_perturb_factors "${speed_perturb_factors}" \
 --pretrained_model $pretrained_model \
 --ignore_init_mismatch true \
 "$@"
