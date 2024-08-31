from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from dotenv import load_dotenv
import os
import torch

# .env ファイルから環境変数をロード
load_dotenv()

# 環境変数を取得
hf_token = os.getenv('HF_TOKEN')

def load_model(device=None, min_duration_on=0.5, min_duration_off=1.0):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # VADモデルとパイプラインの初期化
    segmentation_model = Model.from_pretrained(
    "pyannote/segmentation-3.0", 
    use_auth_token=hf_token)

    vad_model = VoiceActivityDetection(segmentation=segmentation_model)

    HYPER_PARAMETERS = {
        # この値未満の音声区間は無視する
        "min_duration_on": min_duration_on,
        # この値未満の無音区間は音声区間として扱う
        "min_duration_off": min_duration_off
    }
    vad_model.instantiate(HYPER_PARAMETERS)
    vad_model.to(device)
    return vad_model