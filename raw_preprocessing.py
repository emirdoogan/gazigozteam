# raw_preprocessing.py

import cv2
import numpy as np


# Eğitimde kullandığın crop parametreleri
TOP_DIV = 1.30
LEFT_DIV = 2.10
RIGHT_RATIO = 0.93


def crop_raw_image(
    image: np.ndarray,
    top_div: float = TOP_DIV,
    left_div: float = LEFT_DIV,
    right_ratio: float = RIGHT_RATIO
) -> np.ndarray:
    """
    Raw OCT görüntüsünü eğitim pipeline'ındaki yöntemle kırpar:

      - Yükseklik: 0 .. h / top_div
      - Genişlik başlangıcı: w / left_div
      - Genişlik bitişi: w * right_ratio

    Varsayılan:
      top_div = 1.30
      left_div = 2.10
      right_ratio = 0.93
    """
    if image is None:
        raise ValueError("Crop için verilen görüntü None.")

    h, w = image.shape[:2]

    top = int(h / top_div)
    left = int(w / left_div)
    right = int(w * right_ratio)

    # Sınırları görüntü boyutları içinde tut
    top = max(1, min(top, h))
    left = max(0, min(left, w - 1))
    right = max(1, min(right, w))

    if right <= left:
        raise ValueError(
            f"Geçersiz crop sınırları: left={left}, right={right}, "
            f"image_width={w}"
        )

    cropped = image[:top, left:right]

    if cropped.size == 0:
        raise ValueError(
            f"Crop sonucunda boş görüntü oluştu. "
            f"Orijinal boyut={image.shape}, "
            f"crop sınırları=(0:{top}, {left}:{right})"
        )

    return cropped


def preprocess_raw(
    image_path: str,
    target_size: int = None,
    normalize: bool = True
) -> np.ndarray:
    """
    1. Raw OCT görüntüsünü yükler
    2. Gerekirse grayscale'e dönüştürür
    3. crop_raw_image() ile eğitimdeki crop'u uygular
    4. target_size verilmişse kare boyuta resize eder
    5. normalize=True ise görüntüyü [0, 1] aralığına getirir
    6. Kanal boyutu ekler: (H, W, 1)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(
            f"Görüntü okunamadı: {image_path}"
        )

    # Çok kanallı görüntüyü grayscale'e dönüştür
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yeni crop mantığı
    img = crop_raw_image(img)

    # Eğitimde hangi interpolation kullanıldıysa burada da aynı olmalı
    if target_size is not None:
        img = cv2.resize(
            img,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST
        )

    if normalize:
        img = img.astype(np.float32)

        # 8-bit ve 16-bit görüntüleri doğru normalize et
        if img.max() > 0:
            if img.max() <= 255:
                img /= 255.0
            elif img.max() <= 65535:
                img /= 65535.0
            else:
                img /= img.max()

    # (H, W) → (H, W, 1)
    img = np.expand_dims(img, axis=-1)

    return img