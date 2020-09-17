from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class FaceDataset(pl.LightningDataModule):
    def __init__(self, args):
        # dataroot, workers, batch_size, image_size
        self.workers = args.workers
        self.batch_size = args.batch_size

        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.ImageFolder(root=args.dataroot, transform=transform)

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)