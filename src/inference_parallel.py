import datetime
import re
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from speech_to_text import SpeechToText, TranscribeConfig

CUR_DIR = Path(__file__).parent
DATA_DIR = CUR_DIR.parent / "nishika-data"
TEST_CSV_FILE = DATA_DIR / 'test.csv'
SUBMISSION_FILE_NAME = 'submission.csv'
SAMPLE_SUBMISSION_CSV_FILE = DATA_DIR / 'sample_submission.csv'

SUBMISSION_DIR = CUR_DIR.parent / 'submissions'


def remove_spaces(text):
    # 半角スペース(\s)と全角スペース(\u3000)を空文字に置換
    return re.sub(r'[\s\u3000]', '', text)

def main():
    parser = ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--result_csv_file', type=str, default=None)
    parser.add_argument('--split_num', type=int, default=None)
    args = parser.parse_args()
    
    result_csv_file = args.result_csv_file
    processed_row_num = -1
    if result_csv_file is not None:
        result_csv_file = Path(result_csv_file)
        result_df = pd.read_csv(result_csv_file)
        processed_row_num = result_df.index.max()
        submission_dir = result_csv_file.parent
    else:
        datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
        submission_dir = SUBMISSION_DIR / datetime_str
        submission_dir.mkdir(parents=True, exist_ok=True)
    
    submission_file = submission_dir / SUBMISSION_FILE_NAME
    segments_file = submission_dir / 'segments.csv'

    model_dir = args.model_dir
    config_file = args.config_file
    
    stt_config = {}
    if config_file is not None:
        with open(config_file, 'r') as f:
            stt_config = yaml.safe_load(f)
    config = TranscribeConfig(model_dir, **stt_config)
    config_dict = vars(deepcopy(config))
    config_dict.pop('asr_model_dir')
    with open(submission_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config_dict, f)
    
    split_num = args.split_num
    stt = SpeechToText(config)
    
    test_df = pd.read_csv(TEST_CSV_FILE)
    
    # データフレームの長さを取得
    df_length = len(test_df)

    # 8分割するためのステップ数を計算
    step = df_length // 4
    
    # 条件に基づいて適切な部分を抽出
    if 1 <= split_num <= 4:
        start_index = (split_num - 1) * step
        end_index = start_index + step if split_num < 4 else df_length
        split_df = test_df[start_index:end_index]
        print(f"process {len(split_df)} entries.")
    else:
        raise ValueError("split_num should be between 1 and 4")        

    for i, row in split_df.iterrows():
        if i <= processed_row_num:
            print(f"skip {i} entry")
            continue
        
        audio_path = DATA_DIR / row['audio_path']
        stt_results = stt.transcribe(audio_path)
        full_text = stt_results.text
        segments = stt_results.segments
        
        # 全文をCSVに追加
        pd.DataFrame({'ID': [row['ID']], 'target': [full_text]}).to_csv(
            submission_file, 
            mode='a', 
            header=False, 
            index=False
        )
        # セグメントをCSVに追加
        segments_data = [
            {
                'ID': row['ID'],
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            } for segment in segments
        ]
        pd.DataFrame(segments_data).to_csv(
            segments_file, 
            mode='a', 
            header=False, 
            index=False,
        )
        
    df = pd.read_csv(submission_file)
    if len(df) == len(test_df):
        new_row = pd.DataFrame([['ID', 'target']], columns=['ID', 'target'])
        # 新しい行をデータフレームの先頭に追加し、インデックスをリセット
        df = pd.concat([new_row, df]).reset_index(drop=True)
        df["target"] = df["target"].map(remove_spaces)
        df.to_csv(submission_file, index=False, header=False)
        

    print("処理が完了しました。")

if __name__ == '__main__':
    main()