# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import torch

from torch.utils.data import DataLoader,random_split
from loader.loader_mvsec import *
import json


def fetch_dataloader(args):
    """ Create the data loader for the corresponding training/test/demo set """

    # Choose correct config file
    if args.split==1:
        config_path = 'config/split1.json'
    elif args.split==3:
        config_path = 'config/split3.json'
    else:
        raise Exception('Please provide a valid argument for --split. [1/3]')
    
    # Load config file
    config = json.load(open(config_path))

    # Instantiate Dataset
    train_set = Mvsec_Dataset(
                args = config["data_loader"]["train"]["args"],
                type='train',
                path=args.path
            )
    vali_test_set = Mvsec_Dataset(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
    )

    demo_set = Mvsec_Dataset(
                args = config["data_loader"]["demo"]["args"],
                type='demo',
                path=args.path
            )
    
    vali_size = 200
    test_size = len(vali_test_set) - vali_size

    vali_set, test_set = random_split(vali_test_set,[vali_size, test_size],generator=torch.Generator().manual_seed(1234))

    train_loader = DataLoader(train_set,
                                batch_size=config["data_loader"]["train"]["args"]["batch_size"],
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=False)
    vali_loader = DataLoader(vali_set,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    demo_loader = DataLoader(demo_set,batch_size=1,shuffle=True,num_workers=args.num_workers,pin_memory=True)   


    if args.mode == 'train':
        return train_loader,vali_loader,vali_test_set
    
    elif args.mode =='test':
        return test_loader, test_set, vali_test_set
    
    else:
        return demo_loader