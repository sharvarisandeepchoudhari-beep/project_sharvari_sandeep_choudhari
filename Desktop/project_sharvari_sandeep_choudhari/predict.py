import torch
from PIL import Image
from torchvision import transforms
from config import RESIZE_X, RESIZE_Y

transform = transforms.Compose([
    transforms.Resize((RESIZE_X, RESIZE_Y)),
    transforms.ToTensor()
])

def the_predictor(model, image_paths, classes=None):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    if classes is None:
        classes = [str(i) for i in range(10)]

    images = []

    for path in image_paths:
        img = Image.open(path)
        img = transform(img)
        images.append(img)

    images = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    return [classes[p.item()] for p in preds]
