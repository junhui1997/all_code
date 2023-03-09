from __future__ import division
from __future__ import print_function

import utils
import time
import os
import numpy as np
from tcn_train_test import cross_validate
from config import tcn_params, tcn_run_num, result_dir, dataset_name
from config import args

import pdb


def main():

    for run_idx in range(tcn_run_num):

        naming = 'run_{}_{}'.format((run_idx + 1), args.save_name)

        run_result = cross_validate(tcn_params['model_params'], 
                                    tcn_params['train_params'],
                                    naming)  #8x6

        result_file = os.path.join(result_dir, 'tcn_result_{}_split{}.npy'.format(naming, args.split))

        np.save(result_file, run_result)


if __name__ == '__main__':
    main()

