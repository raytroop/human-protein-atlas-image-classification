"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import utils
from model import net, data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def evaluate(model, loss_fn, val_dataloader, metrics, params):
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in val_dataloader:

            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](np.array(output_batch > 0, dtype=np.float32), labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        return metrics_mean


# if __name__ == '__main__':
#     """
#         Evaluate the model on the test set.
#     """
#     # Load the parameters
#     args = parser.parse_args()
#     json_path = os.path.join(args.model_dir, 'params.json')
#     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     params = utils.get_config_from_json(json_path)

#     # use GPU if available
#     params.cuda = torch.cuda.is_available()     # use GPU is available

#     # Set the random seed for reproducible experiments
#     torch.manual_seed(230)
#     if params.cuda:
#         torch.cuda.manual_seed(230)

#     # Get the logger
#     utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

#     # Create the input data pipeline
#     logging.info("Creating the dataset...")

#     # fetch dataloaders
#     dataloaders = data_loader.fetch_dataloader(['val'], args.data_dir, params)
#     test_dl = dataloaders['test']

#     logging.info("- done.")

#     # Define the model
#     model = net.densenet121().cuda() if params.cuda else net.densenet121()

#     loss_fn = nn.BCEWithLogitsLoss()
#     metrics = net.metrics

#     logging.info("Starting evaluation")

#     # Reload weights from the saved file
#     utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

#     # Evaluate
#     test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
#     save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
#     utils.save_dict_to_json(test_metrics, save_path)
