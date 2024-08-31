from argparse import ArgumentParser
from speech_to_text import SpeechToText, TranscribeConfig
from pathlib import Path
import yaml
import pandas as pd
import datetime
import re
from copy import deepcopy

CUR_DIR = Path(__file__).parent
TEST_CSV_FILE = CUR_DIR / 'nishika-data/test.csv'
SUBMISSION_FILE_NAME = 'submission.csv'
SAMPLE_SUBMISSION_CSV_FILE = CUR_DIR / 'sample_submission.csv'

SUBMISSION_DIR = CUR_DIR / 'submissions'


def remove_spaces(text):
    # 半角スペース(\s)と全角スペース(\u3000)を空文字に置換
    return re.sub(r'[\s\u3000]', '', text)


def main():
    parser = ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--result_csv_file', type=str, default=None)
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
    
    stt = SpeechToText(config)
    
    test_df = pd.read_csv(TEST_CSV_FILE)

    for i, row in test_df.iterrows():
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
    
    # post process
    df = pd.read_csv(submission_file)
    df['target'] = df['target'].map(remove_spaces)
    df.to_csv(submission_file, index=False)

    print("処理が完了しました。")

if __name__ == '__main__':
    main()