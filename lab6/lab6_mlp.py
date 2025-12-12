import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

BATCH_SIZE = 50
LEARNING_RATE = 0.09
EPOCHS = 10
HIDDEN_SIZE1 = 512 


TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

print(f"\nРазделение данных: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_size = int(TRAIN_RATIO * len(full_train_dataset))
val_size = int(VAL_RATIO * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nСЛУЧАЙНЫЕ ГИПЕРПАРАМЕТРЫ:")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  LEARNING_RATE: {LEARNING_RATE:.6f}")
print(f"  EPOCHS: {EPOCHS}")
print(f"  HIDDEN_SIZE1: {HIDDEN_SIZE1}")
print(f"\nРазмеры наборов:")

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64,
                 num_classes=10, dropout_rate=0.25):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(hidden_size2, num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, return_probs=False):
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        logits = self.fc_out(x)
        
        if return_probs:
            probs = self.softmax(logits)
            return logits, probs
        
        return logits

model = MLP(
    input_size=28*28,
    hidden_size1=HIDDEN_SIZE1,
    num_classes=10,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                
                val_loss += criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}]:')
        print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Загружены веса лучшей модели (val accuracy: {best_val_accuracy:.2f}%)")
    
    print(f"\nЛучшая точность на валидации: {best_val_accuracy:.2f}%")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits, probs = model(images, return_probs=True)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    return test_accuracy, all_probs, all_labels, all_predictions

print("\n" + "="*50)
print("Начало обучения...")
print("="*50)

train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_validation(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS
)

print("\n" + "="*50)
print("Тестирование...")
print("="*50)

test_accuracy, all_probs, all_labels, all_predictions = test_model(model, test_loader)

def plot_results_with_validation(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy):
    fig = plt.figure(figsize=(14, 8))
    
    gs = fig.add_gridspec(2, 1, height_ratios=[0.4, 1], hspace=0.3)
    
    ax_params = fig.add_subplot(gs[0])
    ax_params.axis('off')
    
    params_text = (
        f"ГИПЕРПАРАМЕТРЫ МОДЕЛИ:\n"
        f"Batch Size: {BATCH_SIZE} | "
        f"Learning Rate: {LEARNING_RATE:.6f} | "
        f"Epochs: {EPOCHS}\n"
        f"Результаты: Train Acc(max)={max(train_accuracies):.1f}% | "
        f"Val Acc(max)={max(val_accuracies):.1f}% | "
        f"Test Acc={test_accuracy:.2f}%"
    )
    
    ax_params.text(0.5, 0.5, params_text, fontsize=10, 
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    gs2 = gs[1].subgridspec(1, 2, wspace=0.3)
    
    ax1 = fig.add_subplot(gs2[0])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Эпоха', fontsize=11)
    ax1.set_ylabel('Потери', fontsize=11)
    ax1.set_title('График потерь при обучении', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs2[1])
    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, 
             label=f'Train (max: {max(train_accuracies):.1f}%)')
    ax2.plot(epochs, val_accuracies, 'r-', linewidth=2,
             label=f'Val (max: {max(val_accuracies):.1f}%)')
    ax2.axhline(y=test_accuracy, color='g', linestyle='--', linewidth=2,
               label=f'Test: {test_accuracy:.1f}%')
    ax2.set_xlabel('Эпоха', fontsize=11)
    ax2.set_ylabel('Точность (%)', fontsize=11)
    ax2.set_title('График точности при обучении', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Многослойный перцептрон для MNIST - Результаты обучения\n'
                f'Тестовая точность: {test_accuracy:.2f}%',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

plot_results_with_validation(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy)

max_train_acc = max(train_accuracies)
max_val_acc = max(val_accuracies)
overfitting_gap = max_train_acc - max_val_acc

print("\n" + "="*50)
print(f"Максимальная точность на тренировке: {max_train_acc:.2f}%")
print(f"Максимальная точность на валидации: {max_val_acc:.2f}%")
print(f"Точность на тесте: {test_accuracy:.2f}%")


import os
from datetime import datetime

save_dir = f"mnist_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_accuracy,
    'params': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'hidden_size1': HIDDEN_SIZE1,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO
    }
}, model_path)

print(f"\nМодель сохранена в: {model_path}")