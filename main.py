"""Console script entry point for ScanNet (``scan-net``).

The thread limits must be set before NumPy, Numba or TensorFlow are imported,
so they are applied here, before ``predict_bindingsites`` -- and with it the
whole scientific stack -- is pulled in.
"""
import os

cores = 5  # Set number of CPUs to use!

for variable in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'NUMBA_DEFAULT_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(variable, '%s' % cores)


def main():
    from predict_bindingsites import main as predict_bindingsites_main
    predict_bindingsites_main()


if __name__ == '__main__':
    main()
