import re
from pathlib import Path

import random
import string


from tqdm import tqdm

cur_dir = Path(__file__).parent

def generate_custom_id():
    # 文字列を大文字に変換し、ランダムな位置に '4' と 'w' を追加
    random_chars = random.choices(string.ascii_letters + string.digits, k=14)
    custom_id = ''.join(random_chars[:6]) + '4' + ''.join(random_chars[6:12]) + 'w' + ''.join(random_chars[12:])
    
    return custom_id

print(generate_custom_id())

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

p_1 = cur_dir / "aozora_work_part1"
p_2 = cur_dir / "aozora_work_part2"
text_files = list(p_1.glob('**/*.txt')) + list(p_2.glob('**/*.txt'))
print(len(text_files))

with open('text_files.txt', 'w') as f:
    for text_file in text_files:
        try:
            f.write(f'{str(text_file).split("/", 2)[-1]}\n')
        except BaseException:
            continue

text_dict = {}
g = open('segments.txt', 'w')
for text_file in tqdm(text_files):
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