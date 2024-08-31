import torch
from pathlib import Path
from espnet2.bin.asr_inference import Speech2Text
from typing import Optional, Dict, Any

def load_model(
    model_dir: str,
    device: Optional[str] = None,
    ctc_weight: float = 0.3,
    lm_weight: float = 0.1,
    maxlenratio: float = 1.0,
    beam_size: int = 20,
    nbest: int = 1,
    **additional_kwargs: Dict[str, Any]
) -> Speech2Text:
    """
    Load ASR model with specified parameters.

    Args:
        model_dir (str): Directory containing model files.
        device (Optional[str]): Device to use for inference. If None, will use CUDA if available, else CPU.
        ctc_weight (float): CTC weight for decoding. Default is 0.3.
        lm_weight (float): Language model weight for decoding. Default is 0.1.
        maxlenratio (float): Maximum length ratio in beam search. Default is 1.0.
        beam_size (int): Beam size for beam search. Default is 20.
        nbest (int): Number of best hypotheses to return. Default is 1.
        **additional_kwargs: Additional keyword arguments to pass to Speech2Text.

    Returns:
        Speech2Text: Loaded ASR model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_dir = Path(model_dir)
    
    # Define file paths
    file_paths = {
        'asr_train_config': model_dir / 'config.yaml',
        'asr_model_file': model_dir / 'asr.pth',
        'lm_train_config': model_dir / 'lm_config.yaml',
        'lm_file': model_dir / 'lm.pth',
    }
    
    # Create initialization parameters
    init_kwargs = {
        **file_paths,
        'ctc_weight': ctc_weight,
        'lm_weight': lm_weight,
        'maxlenratio': maxlenratio,
        'beam_size': beam_size,
        'nbest': nbest,
        'device': device,
        **additional_kwargs  # Include any additional kwargs
    }

    return Speech2Text(**init_kwargs)