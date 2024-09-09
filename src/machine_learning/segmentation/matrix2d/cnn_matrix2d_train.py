import os

from PIL import Image

import torch
from torchvision import models, transforms


# Custom Dataset for loading images and masks
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# Load a pre-trained ResNet101 model
resnet = models.resnet101(pretrained=True)

# Modify the network for semantic segmentation
class SegNet(torch.nn.Module):
    def __init__(self, backbone):
        super(SegNet, self).__init__()
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.conv1x1 = torch.nn.Conv2d(2048, 21, kernel_size=1)  # Adjust number of classes if needed
        self.upsample = torch.nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.upsample(x)
        return x

# Initialize the model, criterion, and optimizer
model = SegNet(resnet).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define data transforms and load datasets
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_data = SegmentationDataset(image_dir='path/to/train_images', mask_dir='path/to/train_masks', transform=transform, mask_transform=mask_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

val_data = SegmentationDataset(image_dir='path/to/val_images', mask_dir='path/to/val_masks', transform=transform, mask_transform=mask_transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)

# Training loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        masks = masks.squeeze(1)  # Ensure masks have the right shape
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

# Evaluation loop
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            masks = masks.squeeze(1)  # Ensure masks have the right shape
            loss = criterion(outputs, masks.long())
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += masks.nelement()
            correct += (predicted == masks).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

print("Training completed")
