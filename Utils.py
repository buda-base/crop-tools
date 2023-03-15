import os
import cv2
import yaml
import numpy as np


def get_img_name(img_path: str) -> str:
    assert os.path.isfile(img_path)

    return os.path.basename(img_path).split(".jpg")[0]


def get_settings(settings_file: str) -> yaml:
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    return settings


def detect_contours(image: np.array, settings: yaml) -> tuple:
    kernel_size = settings["contours"]["kernel"]
    invert = settings["contours"]["invert"]
    iterations = settings["contours"]["iterations"]

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if invert:
        img = cv2.bitwise_not(img)

    img = cv2.dilate(img, kernel, iterations=iterations)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return bw, contours


def crop_image(image: np.array, contours: tuple, settings: yaml) -> list:
    area_threshold = settings["cropping"]["area_threshold"]
    bbox_offset = settings["cropping"]["bbox_offset"]
    cropped_imgs = []

    for cnt in contours:
        area_size = cv2.contourArea(cnt)
        if area_size > (image.shape[0] * image.shape[0]) * area_threshold:
            tmp_mask = np.zeros(
                shape=(image.shape[0], image.shape[1], 1), dtype=np.uint8
            )

            x, y, w, h = cv2.boundingRect(cnt)
            x_offset = w * bbox_offset
            y_offset = h * bbox_offset

            cv2.rectangle(
                tmp_mask,
                (x - int(x_offset), y - int(y_offset)),
                (x + w + int(x_offset), y + h + int(y_offset)),
                (255, 255, 255),
                -1,
            )

            masked_img = cv2.bitwise_and(image, image, mask=tmp_mask)
            masked_img = np.delete(
                masked_img, np.where(~masked_img.any(axis=1))[0], axis=0
            )
            masked_img = np.delete(
                masked_img, np.where(~masked_img.any(axis=0))[0], axis=1
            )

            cropped_imgs.append(masked_img)

    return cropped_imgs


def save_cropped_img(cropped_imgs: list, img_name: str, out_path: str) -> None:
    for idx in range(len(cropped_imgs)):
        out_file = f"{out_path}/{img_name}_{idx}.jpg"
        cv2.imwrite(out_file, cropped_imgs[idx])


def simple_crop(image_path: str, out_path: str, settings: yaml) -> None:
    img = cv2.imread(image_path)
    img_name = get_img_name(image_path)
    bw_img, contours = detect_contours(img, settings=settings)
    cropped_imgs = crop_image(img, contours=contours, settings=settings)
    save_cropped_img(cropped_imgs, img_name=img_name, out_path=out_path)
