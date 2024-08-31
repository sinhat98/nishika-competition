from argparse import ArgumentParser
from speech_to_text import SpeechToText, TranscribeConfig
from pathlib import Path
import yaml
import pandas as pd
import datetime
from copy import deepcopy

CUR_DIR = Path(__file__).parent
DATA_DIR = CUR_DIR.parent / "nishika-data"
TEST_CSV_FILE = DATA_DIR / 'test.csv'
SUBMISSION_FILE_NAME = 'submission.csv'
SAMPLE_SUBMISSION_CSV_FILE = DATA_DIR / 'sample_submission.csv'

SUBMISSION_DIR = CUR_DIR.parent / 'submissions'

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
    
        # 最初の行を書き込むためのフラグ
        first_write = False
        first_write_segments = False
    else:
        datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
        submission_dir = SUBMISSION_DIR / datetime_str
        submission_dir.mkdir(parents=True, exist_ok=True)
        
        # 最初の行を書き込むためのフラグ
        first_write = True
        first_write_segments = True
    
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
        
        audio_path = row['audio_path']
        stt_results = stt.transcribe(audio_path)
        full_text = stt_results.text
        segments = stt_results.segments
        
        # 全文をCSVに追加
        pd.DataFrame({'ID': [row['ID']], 'target': [full_text]}).to_csv(
            submission_file, 
            mode='a', 
            header=first_write, 
            index=False
        )
        first_write = False

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
            header=first_write_segments, 
            index=False
        )
        first_write_segments = False

    print("処理が完了しました。")

if __name__ == '__main__':
    main()