"""Распознавание чисел (цифр) из изображений.

Два режима:
  1) Из файла:   python recognize.py photo.png [photo2.png ...]
  2) Генерация:  python recognize.py --demo   (распознаёт сгенерированные тестовые цифры)

Изображение приводится к 28x28 серому, нормализуется в 0..1 (цифра тёмная на светлом фоне).
"""
import os
import sys
import numpy as np
import torch
from PIL import Image

from model import DigitNet
from data import render_digit

CHECKPOINT = 'digit_net.pt'


def preprocess(img: Image.Image) -> np.ndarray:
    """Приводит PIL-изображение к массиву (1, 1, 28, 28) float32 в [0..1]."""
    g = img.convert('L').resize((28, 28), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    # Предположение: цифра тёмная на светлом фоне. Если так наоборот — инвертируем.
    if arr.mean() > 0.5:
        arr = 255.0 - arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.reshape(1, 1, 28, 28)


def predict(model: DigitNet, x: torch.Tensor, top_k: int = 3):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top = torch.topk(probs, top_k)
    return [(int(i), float(p)) for i, p in zip(top.indices, top.values)]


def load_model(device):
    model = DigitNet().to(device)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f'Файл модели "{CHECKPOINT}" не найден. Сначала запустите: python train.py'
        )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    args = [a for a in sys.argv[1:]]
    if not args or args == ['--demo']:
        # --- Демо: генерируем тестовые цифры и распознаём их ---
        print('--- Демо: распознавание сгенерированных цифр ---')
        np.random.seed(0)
        for digit in range(10):
            img = render_digit(digit, rotate=np.random.uniform(-10, 10),
                               shift=np.random.randint(-2, 3),
                               noise=np.random.uniform(0, 20))
            pil = Image.fromarray(((1.0 - img) * 255).astype(np.uint8))
            x = torch.from_numpy(preprocess(pil)).to(device)
            top = predict(model, x, top_k=3)
            pred = top[0][0]
            conf = top[0][1] * 100
            ok = 'OK ' if pred == digit else 'ERR'
            print(f'[{ok}] Истинная={digit}  Прогноз={pred} (уверенность {conf:.1f}%)   '
                  f'топ: ' + ', '.join(f'{i}:{p*100:.1f}%' for i, p in top))
        return

    if args[0] == '--demo':
        return

    # --- Файловый режим ---
    for path in args:
        if not os.path.exists(path):
            print(f'Не найден файл: {path}')
            continue
        img = Image.open(path)
        x = torch.from_numpy(preprocess(img)).to(device)
        top = predict(model, x, top_k=5)
        print(f'\n{os.path.basename(path)}:')
        for i, p in top:
            mark = ' <-- ' if i == top[0][0] else ''
            print(f'  цифра {i}: {p*100:5.2f}%{mark}')


if __name__ == '__main__':
    main()
