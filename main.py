import json
import argparse
import os
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    cli_args = vars(args)
    args = param.copy()
    for key in ['seed', 'order_seed', 'lambda_sparsity', 'init_logit', 'init_alpha',
                'epochs', 'filepath', 'master_results', 'prefix']:
        if cli_args.get(key) is not None:
            args[key] = cli_args[key]

    # Append seed to filepath so parallel runs with different seeds don't overwrite each other
    base_path = args.get('filepath', './results/')
    if not base_path.endswith('/'):
        base_path += '/'
    seed_suffix = 'S{}_OS{}'.format(args.get('seed', 'NA'), args.get('order_seed', 'NA'))
    args['filepath'] = os.path.join(base_path, seed_suffix) + '/'
    os.makedirs(args['filepath'], exist_ok=True)

    # trainer.py iterates over seed as a list
    if isinstance(args.get('seed'), int):
        args['seed'] = [args['seed']]

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--seed', type=int, help='The seed value')
    parser.add_argument('--order_seed', type=int, help='Seed for class ordering')
    parser.add_argument('--lambda_sparsity', type=float, help='Sparsity regularization weight')
    parser.add_argument('--init_logit', type=float, help='Initial gate logit value')
    parser.add_argument('--init_alpha', type=float, help='Initial alpha value')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--filepath', type=str, help='Path to save results')
    parser.add_argument('--master_results', type=str, help='CSV file to aggregate sweep results')
    parser.add_argument('--prefix', type=str, help='Run prefix for logging')
    # parser.add_argument("--local_rank", type=int, default=0)
    return parser

if __name__ == '__main__':
    main()
