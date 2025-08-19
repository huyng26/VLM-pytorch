from typing import Optional, Dict, List, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
def resize(image:Image.Image, size: Tuple[int, int], resample: Image.Resampling, reducing_gap: Optional[int]= None):
    height, width  = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    ) 
    return resized_image

def rescale(
        image:np.ndarray, scale:float, dtype: np.dtype= np.float32
):
    rescaled_image = image*scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image:  np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]
):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image-mean)/std
    return image

def process_images(
        images,
        size,
        resample: Image.Resampling,
        rescale_factor: float,
        image_mean:Optional[Union[float, List[float]]]= None,
        image_std: Optional[Union[float, List[float]]]=None,

):
    height, width = size[0], size[1]
    image = [
        resize(image=image, size = (height,width), resample = resample) for image in images
    ]
    #convert each images to numpu array
    images = [np.array(image) for image in images]
    #Rescale pixel values to be in the range [0, 1]
    images = [rescale(image, scale = rescale_factor) for image in images]
    #Normalize the images to have the mean of 0 and std of 1
    images = [normalize(image, mean = image_mean, std = image_std) for image in images]  # type: ignore
    images = [image.transpose(2, 0, 1) for image in images] #[Channgel, Height, Width]
    return images


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token* image_seq_len}{bos_token}{prefix_prompt}\n"
class PaliGemmaProcessor:
    #global
    IMAGE_TOKEN = "<image>"
    def __init__(self, tokenizer, image_seq_len: int, image_size: int):
        super().__init__()
        self.image_seq_length = image_seq_len
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # Tokens used for object detection

        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # Tokens for segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        #Adding bos and eos tokens manually
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
        
    def call(self, text: List[str], images: List[Image.Image], truncation: bool= True, padding:bool = True):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"

        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 /255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )

        #Convert the list of numpy array to list of numpy array [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis = 0)
        pixel_values = torch.tensor(pixel_values)

        #Prepend a 'self.image_seq_len" number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length, 
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        #Return the input_ids and attention
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",
            padding=padding,
            truncation=truncation
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data


