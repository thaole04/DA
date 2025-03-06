import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from model.LicensePlateModel import LicensePlateModel

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(np.loadtxt(label_path), dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    # Train process

def main():
    # Set your data path and other parameters
    data_path = 'train'
    input_size = (256, 512)
    num_boxes = 1
    num_epochs = 25
    batch_size = 4
    learning_rate = 0.001

    # Initialize the dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    dataset = LicensePlateDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, criterion, and optimizer
    model = LicensePlateModel(input_size=input_size, num_boxes=num_boxes)
    criterion = nn.MSELoss()  # Assuming the labels are bounding box coordinates
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Train the model
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=num_epochs)

    # Save the model
    torch.save(model.state_dict(), 'license_plate_model.pth')

if __name__ == "__main__":
    main()