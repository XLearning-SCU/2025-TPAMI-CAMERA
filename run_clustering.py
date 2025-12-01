# =====================
# model: Community-aware Multi-view Representation Learning with Incomplete Information
# =====================
# Author: Haobin Li
# Date: Dec, 2025
# E-mail: haobinli.gm@gmail.com,
# =====================
import argparse
import collections
import torch
from model_clustering import CAMERA
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config

dataset = {
    0: "Scene_15",
    1: "LandUse_21",
    2: "NoisyMNIST",
    3: "Reuters",
    4: "MNIST-USPS",
    5: "cub_googlenet",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='0', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--aligned_prop', type=float, default=1.0)  # X A S: ap (1-ap)*ac (1-ap)*(1-ac)
parser.add_argument('--complete_prop', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--rec_lambda', type=float, default=10.0)

args = parser.parse_args()
dataset = dataset[args.dataset]
args.dataset_name=dataset

def main():
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))


    accumulated_metrics = collections.defaultdict(list)
    train_time = 0
    for data_seed in range(0, args.test_time):
        np.random.seed(data_seed)
        random.seed(data_seed)

        train_loader, test_loader, index_loader, class_num = get_dataloader(args=args, aligned_prop=args.aligned_prop, complete_prop=args.complete_prop)

        seed=config['training']['seed']
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Build the model
        model = CAMERA(config)
        
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )

        acc, nmi, ari,test_train_time = model.train(args, config, train_loader, test_loader, index_loader, logger , optimizer, device)
        train_time+=test_train_time
        accumulated_metrics['acc'].append(acc)
        accumulated_metrics['nmi'].append(nmi)
        accumulated_metrics['ari'].append(ari)

    logger.info('--------------------Training over--------------------')
    cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])
    logger.info('******** End, training time = {} s ********'.format(round(train_time, 2)))

if __name__ == '__main__':
    main()
