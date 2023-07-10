import os

from charnet.modeling.model import CharNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from charnet.dataset import CustomDataset
from charnet.config import cfg

# batch_size = 64
batch_size = 1
learning_rate = 0.001
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg.merge_from_file("configs/myconfig.yaml")
cfg.freeze()
print(cfg)

model = CharNet()
# this is how I can load the default weights to train from
model.load_state_dict(torch.load(cfg.WEIGHT))

model.to(device)

# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# data
def calculate_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = model.loss.forward()
    return outputs


# train_dataset = CustomDataset()
all_files = os.listdir("example_samples/images")
num = len(all_files)
train_num = int(float(num) * 0.75)
test_num = num - train_num
train_files, test_files = random_split(all_files, [train_num, test_num])
train_dataset = CustomDataset(train_files, "example_samples/images", "example_samples/labels")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CustomDataset(test_files, "example_samples/images", "example_samples/labels")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data[0].to(device)
        targets = [t.to(device) for t in targets]

        # Vorwärtsdurchlauf
        outputs = model(data)
        loss = calculate_loss(outputs, targets)

        # Rückwärtsdurchlauf und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Ausgabe von Trainingsfortschritt
        if (batch_idx + 1) % 100 == 0:
            print(
                f'Training - Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validierung nach jeder Epoche
    model.eval()  # Setze das Modell in den Evaluationsmodus
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            # Vorwärtsdurchlauf
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Genauigkeit berechnen
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / total

    print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')

print("Training abgeschlossen!")
