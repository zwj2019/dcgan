import argparse

from pytorch_lightning import Trainer
import sys
print(sys.path)
from model.dcgan import DCGAN
from model.data_utils import FaceDataset


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', help="Path to the root of the dataset folder", required=True, type=str)
    parser.add_argument('--workers', help="Number of worker threads for loading the data with the DataLoader", type=int, default=0)
    parser.add_argument('--batch_size', help="Batch size used in training", type=int, default=128)
    parser.add_argument('--image_size', help="Spatial size of the images used for training", type=[int, tuple], default=64)
    parser.add_argument('--nc', help="Number of color channels in the input images", type=int, default=3)
    parser.add_argument('--num_epochs', help="Number of training epochs to run", type=int, default=200)

    # parser.parse_args()
    return parser

if __name__ == '__main__':
    parser = get_argparse()
    parser = DCGAN.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()

    print(args._get_kwargs())

    trainer = Trainer.from_argparse_args(args)
    model = DCGAN(args)