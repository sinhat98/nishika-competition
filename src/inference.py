from argparse import ArgumentParser
from speech_to_text import SpeechToText, TranscribeConfig
from pathlib import Path
import yaml
import polars as pl
import datetime
from copy import deepcopy

CUR_DIR = Path(__file__).parent
DATA_DIR = CUR_DIR.paret / "nishika-data"
TEST_CSV_FILE = DATA_DIR / 'test.csv'
SUBMISSION_FILE_NAME = 'submission.csv'
SAMPLE_SUBMISSION_CSV_FILE = DATA_DIR / 'sample_submission.csv'

SUBMISSION_DIR = CUR_DIR.parent / 'submissions'

def main():
    parser = ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()

    datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
    submission_dir = SUBMISSION_DIR / datetime_str
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_file = submission_dir / SUBMISSION_FILE_NAME
    
    
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
    
    test_df = pl.read_csv(TEST_CSV_FILE)
    # test_dfは以下のようなデータフレーム
    # ID	author	title	daisy_folder	audio_path
    # "06elw1oovcrMbjV"	"fdoFHXEq2E"	"ZW5E4286Nc"	"6Q3jYiU75i"	"test/fdoFHXEq2E/ZW5E4286Nc/6Q3…
    results_df = []
    segment_results_df = []
    for row in test_df.iter_rows(named=True):
        audio_path = row['audio_path']
        stt_results = stt.transcribe(audio_path)
        full_text = stt_results.text
        segments = stt_results.segments
        for segment in segments:
            segment_results_df.append({
                'ID': row['ID'],
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })
        results_df.append({
            'ID': row['ID'],
            'target': full_text
        })
    results_df = pl.DataFrame(results_df)
    segment_results_df = pl.DataFrame(segment_results_df)
    
    results_df.write_csv(submission_file)
    segment_results_df.write_csv(submission_dir / 'segments.csv')
        
            
if __name__ == '__main__':
    main()
