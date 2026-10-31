"""Генерация синтетического датасета изобразительных чисел.

В датасете «The MNIST Database of Handwritten Digits»
каждая цифра представлена изображением 28x28 пикселей.

Мы генерируем изображения чисел, применяя различные
преобразования (сдвиг, поворот, шум) для повышения разнообразия.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import os


def render_digit(digit: int, size: int = 28, font_size: int = 24,
                 shift: int = 0, rotate: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """Возвращает изображение цифры (28x28, значения 0..1)."""
    # Рисуем цифру на белом фоне
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Пытаемся найти подходящий системный шрифт
    font = None
    for name in ('arial.ttf', 'segoeui.ttf', 'calibri.ttf', 'verdana.ttf',
                 'times.ttf', 'cour.ttf', 'impact.ttf'):
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(name, font_size)
            break
        except Exception:
            continue
    if font is None:
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Вычисляем позицию для центрировки
    text = str(digit)
    if font is not None:
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(text, font=font)
    else:
        w, h = 10, font_size

    x = (size - w) // 2 - bbox[0] if font is not None else (size - w) // 2
    y = (size - h) // 2 - (bbox[1] if font is not None else 0)

    draw.text((x + shift, y), text, fill=0, font=font)

    # Поворот
    if rotate:
        img = img.rotate(rotate, fillcolor=255, resample=Image.BILINEAR)

    # Шум
    if noise > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr = np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # Инверсия: цифры тёмные на светлом фоне -> значения 0..1 (1 = пиксель цифры)
    arr = np.asarray(img, dtype=np.float32)
    arr = (255.0 - arr) / 255.0
    return arr


def generate_dataset(num_per_digit: int = 300, seed: int = 42):
    """Возвращает (X, y): X — (N, 1, 28, 28), y — (N,)."""
    np.random.seed(seed)
    xs, ys = [], []
    for digit in range(10):
        for _ in range(num_per_digit):
            shift = np.random.randint(-3, 4)
            rotate = np.random.uniform(-12, 12)
            noise = np.random.uniform(0, 30)
            img = render_digit(digit, shift=shift, rotate=rotate, noise=noise)
            # Небольшая размытия
            if np.random.rand() < 0.3:
                pil = Image.fromarray((img * 255).astype(np.uint8))
                pil = pil.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.3, 0.8))))
                img = np.asarray(pil, dtype=np.float32) / 255.0
                img = 1.0 - (255.0 - img * 255.0) / 255.0  # уже инверсировано, просто нормализуем
            xs.append(img.reshape(1, 28, 28))
            ys.append(digit)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


if __name__ == '__main__':
    # Быстрая демонстрация: рисуем сетку цифр
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for col in range(10):
        for row in range(2):
            img = render_digit(col, rotate=np.random.uniform(-10, 10))
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig('sample_digits.png', dpi=100)
    print('Сохранено: sample_digits.png')
