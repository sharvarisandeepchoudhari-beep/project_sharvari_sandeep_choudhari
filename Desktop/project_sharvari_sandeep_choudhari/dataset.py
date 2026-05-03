from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class TheDataset:
    def __init__(self, path):
        self.data = datasets.ImageFolder(path, transform=transform)

    def get_data(self):
        return self.data

def get_classes(self):
    return self.data.classes

def the_dataloader(dataset, batch_size=4):
    return DataLoader(dataset.get_data(), batch_size=batch_size, shuffle=True)
