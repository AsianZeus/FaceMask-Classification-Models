from EarlyStopping import EarlyStopping
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader
)
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
learning_rate = 1e-3
batch_size = 1024
num_epochs = 20

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.mobilenet_v3_small(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(28224, 100), nn.ReLU(), nn.Linear(100, num_classes)
)
model.to(device)

my_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),
    ]
)

class MaskDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image,y_label)

train_dataset = MaskDataset(
    csv_file="train.csv",
    root_dir="",
    transform=my_transforms,
)

test_dataset = MaskDataset(
    csv_file="test.csv",
    root_dir="",
    transform=my_transforms,

)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, verbose=True
)


train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = [] 

early_stopping = EarlyStopping(patience=4, verbose=True)

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    for data, target in test_loader:
        
        output = model(data)
        
        loss = criterion(output, target)
        
        valid_losses.append(loss.item())
    
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    
    mean_loss = sum(losses) / len(losses)

    scheduler.step(mean_loss)
    print(f"Cost at epoch {epoch+1} is {mean_loss} | valid_loss: {valid_loss:.5f} | train_loss: {train_loss:.5f}")

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
    
    early_stopping(valid_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

def check_accuracy(loader, model):
    print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


def predict(model_path, sample_image):
    model = torch.load(model_path)
    model.eval()
    image = Image.open(sample_image)
    image = my_transforms(image)[None, :, :, :]
    x = model(image)
    return "Mask" if x[0].argmax(dim=0) else "No Mask"


check_accuracy(test_loader, model)
res = predict(model_path="model.pth",sample_image="Mask-Dataset/No_Mask/5.jpg")
print(res)

