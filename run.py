import argparse
import sys

import cv2
import numpy as np

from optimizer import optimizer
from utils import init, enhance


def parse(argv):
    parser = argparse.ArgumentParser(
        description='LightingAdjustment\n' +
                    'Implemented code for Perception-Based Lighting Adjustment of Image Sequences.')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')

    return parser.parse_args(argv)


if __name__ == "__main__":
    # Initial
    args = parse(sys.argv[1:])
    print('Initiating parameter...')
    meta = init(args.input)
    print('Done')

    # Start optimize
    print('Finding optimized parameter...')
    while True:
        prev = np.asarray([meta[k] for k in ['a', 'm', 'f']])
        # Optimize function
        rs = optimizer(**meta)
        diff = np.average(abs(prev - rs))
        # Update variables
        a, m, f = rs
        meta.update({'a': a, 'm': m, 'f': f})
        print('Optimize results (a, m, f): {}\t loss={}'.format([a, m, f], diff))
        # Compare with thresh
        if diff < meta['thresh']:
            print('Optimization finished')
            break
    print('Done')

    # Enhance image
    print('Enhancing image...')
    image_out, prob = enhance(**meta)
    print('Prob H=', prob)

    # Writing result
    cv2.imwrite(args.output, image_out)
    print('Done')
