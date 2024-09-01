import pandas as pd
import argparse
from pathlib import Path

def main(args):
    # データフレームをCSVファイルから読み込む
    df_train = pd.read_csv(args.csv_file)
    df_details = pd.read_csv(args.defails_file)
    df_details = df_details.dropna().reset_index(drop=True)
    
    # DETAIL_IDを4桁の0埋めで変換
    df_details['DETAIL_ID'] = df_details['DETAIL_ID'].map(lambda x: x.split('_')[0] + '_' + x.split('_')[1].zfill(4))
    
    
    id2path = df_train[['ID', 'audio_path']].set_index('ID')['audio_path'].to_dict()
    df_details['audio_path'] = df_details['ID'].map(id2path)
    df_details['audio_path'] = df_details['audio_path'].map(
        lambda x: f"ffmpeg -i {(Path(__file__).parents[1] / 'nishika-data' / x).resolve()} -ar 16000 -ac 1 -f wav - |")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wav.scp
    wav_scp = df_details[['ID', 'audio_path']].drop_duplicates(subset=['ID']).reset_index(drop=True)
    with open(output_dir / 'wav.scp', 'w') as f:
        for _, row in wav_scp.iterrows():
            f.write(row['ID'] + ' ')
            for string in row['audio_path'].split(' '):
                f.write(string)
                if string != '|':
                    f.write(' ')
            f.write('\n')

    # text
    df_details['target_slice'] = df_details['target_slice'].map(lambda x: x.replace(' ', '').replace('"', ''))
    text = df_details[['DETAIL_ID', 'target_slice']]

    text.to_csv(output_dir / 'text', header=False, index=False, sep=' ')

    # utt2spk
    utt2spk = df_details[['DETAIL_ID', 'ID']]
    utt2spk.to_csv(output_dir / 'utt2spk', header=False, index=False, sep=' ')

    # spk2utt
    spk2utt_dict = df_details.groupby('ID')['DETAIL_ID'].apply(list).to_dict()
    with open(output_dir / 'spk2utt', 'w') as f:
        for spk, utts in spk2utt_dict.items():
            f.write(f"{spk} {' '.join(utts)}\n")

    # segments
    df_details['start_time'] = df_details['start_time'].map(lambda x: float(x) / 1000)
    df_details['end_time'] = df_details['end_time'].map(lambda x: float(x) / 1000)
    segments = df_details[['DETAIL_ID', 'ID', 'start_time', 'end_time']]
    segments.to_csv(output_dir / 'segments', header=False, index=False, sep=' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Kaldi format files.")
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('defails_file', type=str, help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save output files')

    args = parser.parse_args()
    main(args)