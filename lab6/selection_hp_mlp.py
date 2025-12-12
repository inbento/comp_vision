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

class MLP(nn.Module):
    def __init__(self, hidden1=256, hidden2=128, dropout=0.5):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def evaluate_params(params, epochs=3):
    """Быстрая оценка параметров"""
    hidden1, hidden2, lr, batch_size, dropout, l2 = params
    
    model = MLP(int(hidden1), int(hidden2), dropout).to(device)
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
    Integer(256, 1024, name='hidden1'),
    Integer(128, 512, name='hidden2'),
    Real(1e-4, 1e-1, name='lr', prior='log-uniform'),
    Integer(32, 256, name='batch_size'),
    Real(0.1, 0.5, name='dropout'), 
    Real(1e-6, 1e-2, name='l2', prior='log-uniform'),
]

print("\n" + "="*50)
print("БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ ДЛЯ MLP")
print("="*50)

result = gp_minimize(
    func=evaluate_params,
    dimensions=space,
    n_calls=20,
    n_random_starts=10,
    random_state=42,
    verbose=True
)

best_params = result.x
print(f"\n🎯 ЛУЧШИЕ ПАРАМЕТРЫ MLP:")
param_names = ['hidden1', 'hidden2', 'lr', 'batch_size', 'dropout', 'l2']
for name, value in zip(param_names, best_params):
    if name in ['hidden1', 'hidden2', 'batch_size']:
        print(f"  {name}: {int(value)}")
    elif name in ['lr', 'l2']:
        print(f"  {name}: {value:.6f}")
    else:
        print(f"  {name}: {value:.3f}")
print(f"  Лучшая точность на валидации: {-result.fun:.2f}%")

print("\n" + "="*50)
print("ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ЛУЧШИХ ПАРАМЕТРАХ")
print("="*50)

final_model = MLP(
    int(best_params[0]), 
    int(best_params[1]), 
    best_params[4]
).to(device)

final_optimizer = optim.Adam(
    final_model.parameters(), 
    lr=best_params[2], 
    weight_decay=best_params[5]
)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=int(best_params[3]), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=int(best_params[3]))
test_loader = DataLoader(test_data, batch_size=int(best_params[3]))

def train_with_history(model, train_loader, val_loader, optimizer, criterion, epochs=15):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
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
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}]:')
        print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print('-' * 50)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Загружены веса лучшей модели (val accuracy: {best_val_accuracy:.2f}%)")
    
    return train_losses, val_losses, train_accs, val_accs, best_val_accuracy

train_losses, val_losses, train_accs, val_accs, best_val_acc = train_with_history(
    final_model, train_loader, val_loader, final_optimizer, criterion, epochs=10
)

print("\n" + "="*50)
print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ НА ТЕСТОВОМ НАБОРЕ")
print("="*50)

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
print(f"Точность на тестовом наборе: {test_accuracy:.2f}%")

def plot_results(train_losses, val_losses, train_accs, val_accs, test_accuracy, best_params):
    fig = plt.figure(figsize=(14, 8))
    
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1], hspace=0.3)
    
    ax_params = fig.add_subplot(gs[0])
    ax_params.axis('off')
    
    params_text = (
        f"ЛУЧШИЕ ПАРАМЕТРЫ MLP (Байесовская оптимизация):\n"
        f"Архитектура: 784 → {int(best_params[0])} → {int(best_params[1])} → 10\n"
        f"Hidden1: {int(best_params[0])} | Hidden2: {int(best_params[1])}\n"
        f"LR: {best_params[2]:.6f} | Batch: {int(best_params[3])} | Dropout: {best_params[4]:.3f} | L2: {best_params[5]:.6f}\n"
        f"Результаты: Val(max)={max(val_accs):.1f}% | Test={test_accuracy:.2f}%"
    )
    
    ax_params.text(0.5, 0.5, params_text, fontsize=10, 
                   verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    gs2 = gs[1].subgridspec(1, 2, wspace=0.3)
    
    ax1 = fig.add_subplot(gs2[0])
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Эпоха', fontsize=11)
    ax1.set_ylabel('Потери', fontsize=11)
    ax1.set_title('График потерь MLP', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs2[1])
    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, 
             label=f'Train (max: {max(train_accs):.1f}%)')
    ax2.plot(epochs_range, val_accs, 'r-', linewidth=2,
             label=f'Val (max: {max(val_accs):.1f}%)')
    ax2.axhline(y=test_accuracy, color='g', linestyle='--', linewidth=2,
               label=f'Test: {test_accuracy:.1f}%')
    ax2.set_xlabel('Эпоха', fontsize=11)
    ax2.set_ylabel('Точность (%)', fontsize=11)
    ax2.set_title('График точности MLP', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'MLP с байесовской оптимизацией на MNIST\n'
                f'Тестовая точность: {test_accuracy:.2f}% | '
                f'Испытаний оптимизации: {len(result.func_vals)}', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

plot_results(train_losses, val_losses, train_accs, val_accs, test_accuracy, best_params)

print("\n" + "="*50)

max_train_acc = max(train_accs)
max_val_acc = max(val_accs)
overfitting_gap = max_train_acc - max_val_acc

print(f"Максимальная точность на тренировке: {max_train_acc:.2f}%")
print(f"Максимальная точность на валидации: {max_val_acc:.2f}%")
print(f"Точность на тесте: {test_accuracy:.2f}%")