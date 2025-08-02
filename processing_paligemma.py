from typing import Optional, Dict, List, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_DEVIATION = [0.5, 0.5, 0.5]
class PaliGemmaProcessor:
    #global

    IMAGE_TOKEN = "<image>"
    def __init__(self, tokenizer, image_seq_len: int, image_size: int):
        super().__init__()
        self.image_seq_len = image_seq_len
        self.image_size = image_size
        