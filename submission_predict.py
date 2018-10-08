import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import utils
from model import net, data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model_name', default='densenet121', help="Model architecture")
parser.add_argument('--best_or_last', default='best',
                    help="Which checkpoint to restore")

def load_model(model_name, checkpoint_path, params, args):
    Net = getattr(net, model_name)
    model = Net(pretrained=False, num_classes=28)
    checkpoint = torch.load(os.path.join(checkpoint_path, '{}.pth.tar'.format(args.best_or_last)))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda() if params.cuda else model
    return model


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.get_config_from_json(json_path)
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dataloaders = dataloaders['test']

    model = load_model(args.model_name, args.model_dir, params, args)

    model.eval()

    out = []
    with torch.no_grad():
        for data_batch in test_dataloaders:
            # move to GPU if available
            if params.cuda: # pylint: disable=E1101
                data_batch = data_batch.cuda(async=True)

            # compute model output
            output_batch = model(data_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            output_batch = (output_batch > 0.0).astype(np.int32)
            for i in range(output_batch.shape[0]):
                output_batch_str = ' '.join(str(v) for v in np.nonzero(output_batch[i])[0].tolist())
                out.append(output_batch_str)

    test_df = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    test_df.Predicted = out
    timestr = time.strftime("%m%d-%H%M")
    test_df.to_csv('submission{}_{}.csv'.format(timestr, args.best_or_last), index=False)
