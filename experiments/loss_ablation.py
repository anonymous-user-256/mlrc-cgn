from experiment_utils import set_env
set_env()

from cgn_framework.imagenet import train_cgn, config

import argparse

def disable_loss_from_config(cfg):
    """Disable the losses as specified in by the configuration of the experiment."""
    if 'shape' in cfg.disable_loss:
        cfg.LAMBDA.BINARY = 0
        cfg.LAMBDA.MASK = 0
    if 'text' in cfg.disable_loss:
        cfg.LAMBDA.TEXT = [0, 0, 0, 0]
    if 'background' in cfg.disable_loss:
        cfg.LAMBDA.BG = 0
    if 'reconstruction' in cfg.disable_loss:
        cfg.LAMBDA.L1 = 0
        cfg.LAMBDA.PERC = [0, 0, 0, 0]

def main(args):
    cfg = config.get_cfg_defaults()
    cfg = train_cgn.merge_args_and_cfg(args, cfg)
    cfg.disable_loss = args.disable_loss

    print(cfg)
    disable_loss_from_config(cfg)

    train_cgn.main(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments from original training script
    parser.add_argument('--model_name', default='tmp',
                        help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument('--weights_path', default='',
                        help='provide path to continue training')
    parser.add_argument('--sampled_fixed_noise', default=False, action='store_true',
                        help='If you want a different noise vector than provided in the repo')
    parser.add_argument('--save_singles', default=False, action='store_true',
                        help='Save single images instead of sheets')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for noise sampling')
    parser.add_argument('--episodes', type=int, default=50,
                        help="We don't do dataloading, hence, one episode = one gradient update.")
    parser.add_argument('--batch_sz', type=int, default=1,
                        help='Batch size, use in conjunciton with batch_acc')
    parser.add_argument('--batch_acc', type=int, default=2048,
                        help='pseudo_batch_size = batch_acc*batch size')
    parser.add_argument('--save_iter', type=int, default=2048,
                        help='Save samples/weights every n iter')
    parser.add_argument('--log_losses', default=False, action='store_true',
                        help='Print out losses')
    # Add new argument for that disables specific losses
    parser.add_argument('--disable_loss', type=str, nargs='*', default=[],
                        choices=['shape', 'text', 'background', 'reconstruction'],
                        help='Choose 0 or more losses whose weight will become 0')
    args = parser.parse_args()

    main(args)