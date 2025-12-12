import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

print(f"Размеры: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_data)}")

class CNN(nn.Module):
    def __init__(self, conv1=32, conv2=64, fc1=128, dropout=0.5):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, conv1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1, conv2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(conv2*7*7, fc1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc1, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def evaluate_params(params, epochs=4):
    """Быстрая оценка параметров"""
    conv1, conv2, fc1, lr, batch_size, dropout, l2 = params
    
    model = CNN(int(conv1), int(conv2), int(fc1), dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size))
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    return -val_accuracy

space = [
    Integer(16, 128, name='conv1'),
    Integer(32, 256, name='conv2'),    
    Integer(64, 512, name='fc1'),  
    Real(1e-4, 1e-1, name='lr', prior='log-uniform'),
    Integer(32, 256, name='batch_size'),
    Real(0.1, 0.5, name='dropout'),
    Real(1e-6, 1e-2, name='l2', prior='log-uniform'),
]

print("\nЗапуск байесовской оптимизации...")
result = gp_minimize(
    func=evaluate_params,
    dimensions=space,
    n_calls=20,
    n_random_starts=5,
    random_state=42,
    verbose=True
)

best_params = result.x
print(f"\n🎯 Лучшие параметры:")
param_names = ['conv1', 'conv2', 'fc1', 'lr', 'batch_size', 'dropout', 'l2']
for name, value in zip(param_names, best_params):
    if name in ['conv1', 'conv2', 'fc1', 'batch_size']:
        print(f"  {name}: {int(value)}")
    elif name in ['lr', 'l2']:
        print(f"  {name}: {value:.6f}")
    else:
        print(f"  {name}: {value:.3f}")
print(f"  Лучшая точность на валидации: {-result.fun:.2f}%")

print("\nФинальное обучение на лучших параметрах...")
final_model = CNN(
    int(best_params[0]), 
    int(best_params[1]), 
    int(best_params[2]), 
    best_params[5]
).to(device)

final_optimizer = optim.Adam(
    final_model.parameters(), 
    lr=best_params[3], 
    weight_decay=best_params[6]
)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=int(best_params[4]), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=int(best_params[4]))
test_loader = DataLoader(test_data, batch_size=int(best_params[4]))

def train_with_history(model, train_loader, val_loader, optimizer, criterion, epochs=15):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

train_losses, val_losses, train_accs, val_accs = train_with_history(
    final_model, train_loader, val_loader, final_optimizer, criterion, epochs=10
)

final_model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = final_model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"\n📊 Точность на тестовом наборе: {test_accuracy:.2f}%")

fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(12, 8))

epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Потери')
ax1.set_title('График потерь')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-', label=f'Train (max: {max(train_accs):.1f}%)')
ax2.plot(epochs_range, val_accs, 'r-', label=f'Val (max: {max(val_accs):.1f}%)')
ax2.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test: {test_accuracy:.2f}%')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Точность (%)')
ax2.set_title('График точности')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'CNN на MNIST\nТестовая точность: {test_accuracy:.2f}%', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Лучшая точность на тесте: {test_accuracy:.2f}%")