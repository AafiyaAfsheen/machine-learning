import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data_iter = iter(train_loader)
images, labels = next(data_iter)

images = images / 2 + 0.5  
np_images = images.numpy()

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax = axes[i]
    ax.imshow(np_images[i][0], cmap='gray')
    ax.set_title(classes[labels[i]])
    ax.axis('off')

plt.tight_layout()
plt.show()
