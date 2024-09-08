# Nishika

## 内容物
zipファイルを解凍すると以下のようなディレクトリ構造になる。

```bash
.
├── README.md
├── entry_points.md
├── aozora
├── config
├── models
├── nishika-recipe
├── reazonspeech-espnet-v2
├── pyproject.toml
├── requirements.txt
├── run.sh
└── src
```


## モデルの学習
モデルの学習はespnetを用いて行う。


### espnetの環境構築
```bash
git clone https://github.com/espnet/espnet.git -b v.202402
cd espnet/tools
./setup_anaconda.sh miniconda espnet 3.10
make
. ./activate_python.sh
```

### nishika用のレシピ作成

#### 青空文庫のテキストデータをダウンロード
※最終生成物lm_train.txtは生成済みのものを同封しているのでこの作業はskipできます。

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
これを実行することにより以下のような`lm_train.txt`が作成される。
```bash
LfWDrG4n8FK8Twxq_0000 詩というものについて、
LfWDrG4n8FK8Twxq_0001 ただに詩についてばかりではない。
LfWDrG4n8FK8Twxq_0002 私の今日まで歩いてきた路は、
LfWDrG4n8FK8Twxq_0003 ちょうど手に持っている蝋燭の蝋のみるみる減っていくように、
LfWDrG4n8FK8Twxq_0004 生活というものの威力のために
```


`lm_train.txt`を作る際に使用したテキストファイルの一覧は以下の通りです。<br>
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

## 推論(提出物の作成)
### 環境構築

```bash
# pythonの環境構築 (python環境が用意されていない場合)
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.10.14
pyenv local 3.10.14

# python環境がすでに用意されている場合
pip install requirements.txt
```

[pyannote-audioのsegmentationモデル](https://huggingface.co/pyannote/segmentation-3.0)を使うために.envファイルにhuggingfaceのアクセストークンを設定します。

アクセストークンの詳細については[こちら](https://huggingface.co/docs/hub/security-tokens)を参照してください。

#### 推論コードの実行
```bash
# GPU1枚で推論させる場合(2日程度かかります)
poetry run python src/inference_with_step_log.py models --config_file conf/best_decode_config.yaml
```

```bash
# GPUを4枚使用する場合(12h程度で完了します)
bash run.sh
```

これによりsubmissionsに以下のような形で提出ファイルが作成される
```bash
├── yyyyMMddHHmm(日時)
    ├── config.yaml
    ├── segments.csv
    └── submission.csv
```