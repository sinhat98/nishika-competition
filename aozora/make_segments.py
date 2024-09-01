import re
from pathlib import Path

import random
import string

from tqdm import tqdm

cur_dir = Path(__file__).parent

def load_file_list():
    with open(cur_dir / "file_list.txt") as f:
        file_list = f.readlines()
    return file_list

def load_all_files():
    p1 = cur_dir / "aozora_work_part1"
    p2 = cur_dir / "aozora_work_part2"
    file_list = list(p1.glob('**/*.txt')) + list(p2.glob('**/*.txt'))
    return file_list

def generate_custom_id():
    # 文字列を大文字に変換し、ランダムな位置に '4' と 'w' を追加
    random_chars = random.choices(string.ascii_letters + string.digits, k=14)
    custom_id = ''.join(random_chars[:6]) + '4' + ''.join(random_chars[6:12]) + 'w' + ''.join(random_chars[12:])
    
    return custom_id

def normalize_text(text):
    text = re.sub(r'――', ' ', text)
    text = re.sub(r'「', ' ', text)
    text = re.sub(r'」', ' ', text)
    text = re.sub(r'……', ' ', text)
    text = re.sub(r'\?', '', text)
    text = re.sub(r' ', '', text)
    # ()を削除
    text = re.sub(r'\(\)', '', text)
    # 〜を削除
    text = re.sub(r'〜', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

text_dict = {}
file_list = load_all_files()
print(len(file_list))
with open('file_list.txt', 'w') as f:
    for text_file in file_list:
        f.write(str(text_file) + '\n')

g = open('lm_train.txt', 'w')
for text_file in tqdm(file_list):
    if isinstance(text_file, str):
        text_file = cur_dir / text_file.strip()
    fileid = generate_custom_id()
    with open(text_file, 'r') as f:
        lines = f.readlines()
        c = 0
        for line in lines:
            
            line = line.strip()
            # if line.startswith('行番号'):
            #     fileid = line.split('\t')[-1].split('.')[0]
            if line.endswith('[青空文庫テキスト]'):
                #text_dict[fileid] = normalize_text(line.split('\t')[0])
                text = normalize_text(line.split('\t')[0])
                if len(text) == 0:
                    continue
                if len(text) == 1 and not re.match(r'[〇一二三四五六七八九十百千万億兆]', text):
                    continue
                fileid_all = fileid + f'_{c:04d}' 
                c += 1
                g.write(f'{fileid_all} {text}\n')
                
g.close()