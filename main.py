import argparse, os
import yaml

from config import ConfigLoader
import src.sessions.utr5_unet1dmodel as utr5_unet1dmodel
import src.sessions.utr5_vae as utr5_vae

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='utr5_vae', help='model to train/sweep/generate')
    parser.add_argument('--oper', type=str, default='train', help='trian or sweep or generate.')
    
    return parser

def prepare_config(path, root):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    config = ConfigLoader(cfg, root)
    return config

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    if args.oper == 'train':
        cfg_path = root + '/src/config/train_cfgs/'
        cfg_file = cfg_path + args.model + '.yaml'
        config = prepare_config(cfg_file, root)

        run = eval(args.model).train
    elif args.oper == 'sweep':
        # TODO
        cfg_path = root + '/src/config/sweep_cfgs/'
    elif args.oper == 'generate':
        #TODO
        cfg_path = root + '/src/config/generate_cfgs/'
    else:
        raise Exception("not supported operation, should be one of [train, sweep, generate]")
        
    run(config)





