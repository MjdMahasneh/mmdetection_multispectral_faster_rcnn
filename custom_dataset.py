import torch
from torch.utils.data import Dataset




class CustomDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images  # List of 7-channel images (as tensors)
        self.annotations = annotations  # List of dictionaries with 'boxes' and 'labels'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.annotations[idx]



if __name__ == "__main__":

    # Example data (replace with real data)
    images = [torch.randn(7, 256, 256) for _ in range(10)]  # 7-channel random images
    annotations = [
        {"boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
         "labels": torch.tensor([1], dtype=torch.int64)}
        for _ in range(10)
    ]


    # Create a dataset
    dataset = CustomDataset(images, annotations)