"""Peform hyperparemeters search"""

import argparse
import os
import subprocess
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/image_size',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    utils.save_dict_to_json(params, json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.get_config_from_json(json_path)

    # Perform hypersearch over one parameter
    # image_size = [128, 224, 512, 1024]
    # batch_size = [64, 32, 16, 8]
    image_size = [1024]
    batch_size = [4]
    for sz, bz in zip(image_size, batch_size):
        # Modify the relevant parameter in params
        params.image_size = sz
        params.batch_size = bz

        # Launch job (name has to be unique)
        job_name = "image_size_{}_bz{}".format(sz, bz)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
