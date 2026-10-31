"""Нейросетевая модель для распознавания чисел (цифр 0-9)."""
import torch
import torch.nn as nn


class DigitNet(nn.Module):
    """Небольшие convolutional neural network для классификации 28x28 изображений цифр.

    Архитектура:
      Conv(1->16) -> ReLU -> MaxPool
      Conv(16->32) -> ReLU -> MaxPool
      Flatten
      Linear(32*7*7 -> 128) -> ReLU -> Dropout(0.3)
      Linear(128 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = DigitNet()
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print('Вход: ', tuple(x.shape))
    print('Вывод:', tuple(y.shape))
    print('Параметры:', sum(p.numel() for p in model.parameters()))
