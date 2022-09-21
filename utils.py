import math

import cv2
import numpy as np
from PIL import Image, ImageFilter

from functions import HVS_func, entropy_H


def init(image_path):
    """
    Initiate params for the optimizing step
    :param image_path:  Input image path
    :return:            Initiated params (m, f, x_glob_a, x_local_a, N)
    """
    image = cv2.imread(image_path)

    # Get channel data
    I_b, I_g, I_r = np.transpose(image, (2, 0, 1))

    # Calculate luminance of image
    L = 0.2125 * I_r + 0.7154 * I_g + 0.0721 * I_b
    L_max = np.max(L)
    L_min = np.min(L)
    L_av = np.average(L)

    # Calculate key of image
    k = (L_max - L_av) / (L_max - L_min)

    # Calculate the parameter m based on the key value
    m = .3 + .7 * (k ** 1.4)
    f = 1

    # Calculate the adaption level
    x_glob_a = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # mean grey level
    x_pil = Image.fromarray(image)
    x_local_a = x_pil.filter(ImageFilter.GaussianBlur)  # low-frequency information -> Gauss kernel
    x_local_a = np.asarray(x_local_a)

    # Calculate optimal N based on the perfect entropy (H=-8)
    # When H=-8, the probability of pixel values must be equal 1/256 (from pixel 0->255)
    # Thus, p(i)=S/256 with S is the number of pixels in the image
    h, w, c = image.shape
    N = h * w / 256
    N = sum(float(i) * N for i in range(256)) * 3

    a = np.random.random()

    # Thresholds
    thresh_a = np.asarray([[0, 1]])
    thresh_mf = np.asarray([[0, 3.1 - 1e-8], [math.exp(-8), math.exp(8)]])
    thresh = 1e-8

    # Gather params
    meta = {'a': a, 'm': m, 'f': f, 'N': N, 'image': image,
            'x_glob_a': x_glob_a, 'x_local_a': x_local_a,
            'thresh_a': thresh_a, 'thresh_mf': thresh_mf, 'thresh': thresh}
    return meta


def enhance(image, a, m, f, x_glob_a, x_local_a, **kwargs):
    """
    Enhance image using HVS function
    :param image:   Input image
    :return:        Enhanced image, Pixel distribution probability
    """
    y = HVS_func(image, a=a, m=m, f=f, x_glob_a=x_glob_a, x_local_a=x_local_a)
    y = np.clip(y.astype(np.uint8), 0, 255)
    y_h = entropy_H(y)
    return y, y_h
