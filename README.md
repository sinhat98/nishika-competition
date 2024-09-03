# Nishika

## Train

### espnetの環境構築
```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools
./setup_anaconda.sh miniconda espnet 3.10
make
. ./activate_python.sh
```

### nishika用のレシピ作成
#### データのダウンロード
##### コンペ用データをダウンロード
[こちら](https://competition.nishika.com/competitions/audio_book_transcription/data)からnishikaのコンペデータをダウンロードしてnishika-dataディレクトリに配置する。
```bash
cd nishika-data
unzip train.zip test.zip
```
最終的に以下の構造になる。
```
nishika-data/
├── data_explanation.xlsx
├── test.csv
├── test.zip
├── train.csv
├── train.zip
└── train_details.csv
```

##### 青空文庫のテキストデータをダウンロード
ルートディレクトリに移動した以下を実行
zipの解凍方法は最終的に文字化けせずに解凍できればどんな方法でも構いません。
```bash
cd ../../aozora
wget https://lab.ndl.go.jp/dataset/hurigana-speech-corpus-aozora/aozora_work_part1.zip
wget https://lab.ndl.go.jp/dataset/hurigana-speech-corpus-aozora/aozora_work_part2.zip
unzip aozora_work_part1.zip
unzip aozora_work_part2.zip
sudo apt-get install convmv
convmv -f shift-jis -t utf-8 --notest aozora_work_part1/*.zip # 文字化け対応
convmv -f shift-jis -t utf-8 --notest aozora_work_part2/*.zip # 文字化け対応
python unzip_all.py aozora_work_part1
python unzip_all.py aozora_work_part2
# 必要に応じて
# python format_dirname.py aozora_work_part1
# python format_dirname.py aozora_work_part2
python make_lm_train_text.py
cd ..
```
これを実行することにより以下のような`segments.txt`が作成される。
```bash
LfWDrG4n8FK8Twxq_0000 詩というものについて、
LfWDrG4n8FK8Twxq_0001 ただに詩についてばかりではない。
LfWDrG4n8FK8Twxq_0002 私の今日まで歩いてきた路は、
LfWDrG4n8FK8Twxq_0003 ちょうど手に持っている蝋燭の蝋のみるみる減っていくように、
LfWDrG4n8FK8Twxq_0004 生活というものの威力のために
```


[実際に訓練に用いたテキストファイル名一覧](aozora/file_list.txt)


#### レシピディレクトリ作成
```bash
root=$(pwd)
espnet_dir="${root}/espnet/egs2/nishika/asr1"

# nishikaレシピを作成
cd espnet/egs2
task=asr1
TEMPLATE/${task}/setup.sh nishika/${task}
cp ${root}
cd nishika-recipe
cp run.sh $espnet_dir
cp conf/* "${espnet_dir}/conf/"
cp local/* "${espnet_dir}/local/"
cd ..

# kaldi-styleのディレクトリ作成
. espnet/tools/activate_python.sh
cd $espnet_dir
./run.sh --stop-stage 5 # オーディオデータとトークナイザの構築

# 言語モデルの訓練用ファイル作成
cd $root
mv ${espnet_dir}/dump/raw/lm_train.txt ${espnet_dir}/dump/raw/lm_train_tmp.txt
cat ${espnet_dir}/dump/raw/lm_train_tmp.txt aozora/lm_train.txt > espnet/egs2/nishika/asr1/dump/raw/lm_train.txt
```

#### 事前学習モデル(ReazonSpeech)をダウンロード
```bash
espnet_dir="espnet/egs2/nishika/asr1"
# git-lfsをインストール
sudo apt-get install git-lfs
git lfs install

# huggingfaceのリポジトリをクローン
git clone https://huggingface.co/reazon-research/reazonspeech-espnet-v2 "${espnet_dir}/reazonspeech-espnet-v2"
```


#### 訓練scriptの実行
```bash
cd espnet/egs2/nishika/asr1

./run.sh --stage 6 --stop-stage 7 # 言語モデルの訓練

./run.sh --stage 10 --stop-stage 11 # 音声認識モデルの訓練

```

正常に訓練が終了すると以下のように`exp`ディレクトリが出来上がる
```
exp
├── asr_stats_raw_jp_char_sp
├── asr_train_asr_reazon_ft_raw_jp_char_sp
├── lm_stats_jp_char
└── lm_train_lm_reazon_jp_char
```

## Inference
### 環境構築
何らかのpython環境を構築済みであることを仮定

```bash
pip install poetry
poetry lock
poetry install
```

### モデルの準備

#### expディレクトリから作成
```bash
espnet_exp_dir=espnet/egs2/nishika/asr1/exp
mkdir models
cp -r "${espnet_exp_dir}/asr_stats_raw_jp_char_sp" models/asr_stats_raw_jp_char_sp
cp -r "${espnet_exp_dir}/lm_stats_jp_char" models/lm_stats_jp_char
cp -r "${espnet_exp_dir}/asr_stats_raw_jp_char_sp" models/asr_stats_raw_jp_char_sp
cp "${espnet_exp_dir}/asr_train_asr_reazon_ft_raw_jp_char_sp/valid.acc.ave_3best.pth" models/asr.pth
cp "${espnet_exp_dir}/lm_train_lm_reazon_jp_char/valid.loss.ave_1best.pth" models/lm.pth
```

#### huggingface hubから訓練済みモデルをダウンロード
`sinhat98/nishika-competition`にモデルをアップロード済みなのでここからダウンロード

```bash
poetry run huggingface-cli login # tokenを入力してログイン
HF_HUB_ENABLE_HF_TRANSFER=1 poetry run huggingface-cli download sinhat98/nishika-competition # リポジトリをダウンロード
```

#### 提出してmodels.zipを用いる場合
models.zipをルートディレクトリ配下に配置して解凍してください。


### 推論コードの実行
#### T4 1枚で推論

```bash
poetry run python src/inference_with_step_log.py models --config_file conf/best_decode_config.yaml
```

#### T4を4枚用いる場合

```bash
bash run.sh
```