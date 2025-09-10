import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .SHA_origin import build as build_sha_origin

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
        # return build_sha_origin(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
