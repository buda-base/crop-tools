"""
A very simple script based on solely on contour detection
"""


import os
import cv2
import random
import yaml
import numpy as np
from tqdm import tqdm
from glob import glob
from Utils import simple_crop, get_settings


data_root = "Data"
test_data = ["W0ULDC328154", "W0LULDC348658"]
settings = "Settings\crop_settings1.yaml"


def run_simple_cropping(data_set: str, settings_file: str):
    out_path = os.path.join(data_set, "cropped")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = glob(f"{data_set}/*.jpg")
    crop_settings = get_settings(settings_file=settings_file)

    for idx in tqdm(range(len(images))):
        simple_crop(images[idx], out_path=out_path, settings=crop_settings)


if __name__ == "__main__":
    for ts in test_data:
        run_simple_cropping(
            data_set=os.path.join(data_root, ts), settings_file=settings
        )
