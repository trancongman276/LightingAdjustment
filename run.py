import math
import cv2
import matplotlib
import numpy as np

from PIL import Image, ImageFilter
from scipy.optimize import minimize

matplotlib.use('TkAgg')


def HVS_func(X, a, f, m, x_glob_a, x_local_a, V_max=255):
    """
    **Implementation of Human Vision System**\n
    The parameter a controls the contribution of global adaptation level (*x_glob_a*) and local adaptation level (*x_local_a*)\n
    The image contrast is enhanced when *m* increased\n
    The image brightness is enhanced when *f* is decreasing

    :param X:       Input image
    :param a:       Parameter a (for adaptation level)
    :param f:       Parameter f (for brightness)
    :param m:       Parameter m (for contrast)
    :param V_max:   Maximum range of output value
    :return:        Enhanced image
    """
    assert 0 <= a <= 1, "a must be in range [0, 1]"
    assert 0 <= m < 3.1, "m must be in range [0, 3.1)"
    assert math.exp(-8) <= f <= math.exp(8), "f must be in range [exp(-8), exp(8)]"

    # Calculate the adaptation level
    x_a = a * x_local_a + (1 - a) * x_glob_a

    # Calculate the semi-saturation constant
    ssc = (f * x_a) ** m
    rs = (X / (X + ssc + 1e-9)) * V_max
    return rs


def init(image):
    """
    Initiate params for the optimizing step
    :param image: Input image
    :return:      Initiated params (m, f, x_glob_a, x_local_a, N)
    """
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
    return m, f, x_glob_a, x_local_a, N


def entropy_H(X):
    """
    Calculate the pixels probability in given image
    :param X: Input image
    :return:  Probability
    """
    H = 0
    w, h, c = X.shape
    total_pixel = w * h * c
    values, counts = np.unique(X, return_counts=True)
    for c in counts:
        p = c / total_pixel
        H += p * math.log2(p)
    H = -H
    return H


def fun(img, a, m, f, x_glob_a, x_local_a, N):
    """
    Simplex function to be optimized

    :return:    The different between the "perfect" brightness image with the current one
    """
    _img = img.copy()
    y = HVS_func(_img, a=a, m=m, f=f, x_glob_a=x_glob_a, x_local_a=x_local_a)
    return abs(y.sum() - N)


def optimizer(a, m, f,
              thresh_a, thresh_mf,
              method="Nelder-Mead",):
    """
    Optimize parameters using "Nelder-Mead" method which is based on "Simplex method"
    :return:    Optimized parameters
    """

    alpha = minimize(lambda x: fun(image, a=x, m=m, f=f, x_glob_a=x_glob_a, x_local_a=x_local_a, N=N),
                     x0=a,
                     method=method,
                     bounds=thresh_a)
    a = alpha.get('x')[0]
    mf = minimize(lambda x: fun(image, a=a, m=x[0], f=x[1], x_glob_a=x_glob_a, x_local_a=x_local_a, N=N),
                  x0=np.asarray([m, f]),
                  method=method,
                  bounds=thresh_mf)
    m, f = mf.get('x')
    return np.asarray([a, m, f])


if __name__ == "__main__":
    # Initial
    print('Initiating parameter...')
    image = cv2.imread('./3.jpg')
    m, f, x_glob_a, x_local_a, N = init(image)
    a = np.random.random()

    # Thresholds
    thresh_a = np.asarray([[0, 1]])
    thresh_mf = np.asarray([[0, 3.1 - 1e-8], [math.exp(-8), math.exp(8)]])
    thresh = 1e-8
    print('Done')

    # Start optimize
    print('Finding optimized parameter...')
    while True:
        prev = np.asarray([a, m, f])
        # Optimize function
        rs = optimizer(*prev, thresh_a=thresh_a, thresh_mf=thresh_mf)
        diff = np.average(abs(prev - rs))
        print('Optimize results (a, m, f): {}\t loss={}'.format([a, m, f], diff))
        if diff < thresh:
            print('Optimization finished')
            break
        a, m, f = rs
    print('Done')

    # Enhance image
    print('Enhancing image...')
    y = HVS_func(image, a=a, m=m, f=f, x_glob_a=x_glob_a, x_local_a=x_local_a)
    y = np.clip(y.astype(np.uint8), 0, 255)
    y_h = entropy_H(y)
    print('Prob H=', y_h)
    # Writing result
    cv2.imwrite('result.jpg', y)
    # plt.imshow(y,), plt.show()
    cv2.imshow('test', y)
    cv2.waitKey(0)
    print('Done')
