from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
import numpy as np
from torch.utils import data

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}

def CreateSrcDataLoader(args): 
    if args.source == 'gta5':
        source_dataset = GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['gta5'], mean=IMG_MEAN)
    elif args.source == 'synthia':
        source_dataset = SYNDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                                    crop_size=image_sizes['synthia'], mean=IMG_MEAN)  
    else:
        raise ValueError('The target dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    return source_dataloader

def CreateTrgDataLoader(args):
    if args.data_label_folder_target is not None:
        target_dataset = cityscapesDataSetLabel(args.data_dir_target, args.data_list_target,
                                           max_iters=args.num_steps * args.batch_size,
                                           crop_size=image_sizes['cityscapes'], mean=IMG_MEAN,
                                           set=args.set, label_folder=args.data_label_folder_target) 
    else:
        if args.set == 'train':
            target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                               max_iters=args.num_steps * args.batch_size,
                                               crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set) 
        else:
            target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set)             

    if args.set == 'train':
        target_dataloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    return target_dataloader

def CreateTrgDataSSLLoader(args):
    target_dataset = cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                           crop_size=image_sizes['cityscapes'], mean=IMG_MEAN, set=args.set)
    target_dataloader = data.DataLoader(target_dataset, batch_size=1, shuffle=False, pin_memory=True)  
    return target_dataloader
