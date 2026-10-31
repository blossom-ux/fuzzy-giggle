"""Обучение сети распознавания чисел."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data import generate_dataset
from model import DigitNet

EPOCHS = 20
BATCH = 128
LR = 1e-3
DATA_SIZE = 300          # изображений на каждую цифру
CHECKPOINT = 'digit_net.pt'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Устройство:', device)

    # --- Данные ---
    X, y = generate_dataset(num_per_digit=DATA_SIZE)
    print(f'Датасет: X={X.shape}, y={y.shape}')

    # 80/20 train/val
    n = X.shape[0]
    idx = torch.randperm(n)
    cut = int(0.8 * n)
    train_idx, val_idx = idx[:cut], idx[cut:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    # --- Модель ---
    model = DigitNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # --- Обучение ---
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        # --- Валидация ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total += xb.size(0)

        train_acc = 100.0 * correct / total
        val_acc = 100.0 * v_correct / v_total
        train_loss = loss_sum / total
        print(f'Эпоха {epoch:2d}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f'  -> сохранена лучшая модель (val_acc={val_acc:.2f}%)')

    print(f'\nЛучшая валидация: {best_val_acc:.2f}%')
    print(f'Модель сохранена: {os.path.abspath(CHECKPOINT)}')


if __name__ == '__main__':
    main()
