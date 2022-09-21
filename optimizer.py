import numpy as np
from scipy.optimize import minimize

from functions import HVS_func


def fun(img, a, m, f, x_glob_a, x_local_a, N, **kwargs):
    """
    Simplex function to be optimized

    :return:    The different between the "perfect" brightness image with the current one
    """
    _img = img.copy()
    y = HVS_func(_img, a=a, m=m, f=f, x_glob_a=x_glob_a, x_local_a=x_local_a)
    return abs(y.sum() - N)


def optimizer(image,
              a, m, f, N,
              x_glob_a, x_local_a,
              thresh_a, thresh_mf,
              method="Nelder-Mead",
              **kwargs):
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

