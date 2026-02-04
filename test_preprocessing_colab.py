# ============================================================
# TEST PREPROCESAMIENTO - SOLO VISUALIZACIÓN
# Dataset: ZIP con carpetas de clases
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from collections import defaultdict

# ============================================================
# PREPROCESADOR
# ============================================================

class ImagePreprocessor:
    def __init__(self, out_size=(256, 256)):
        self.out_size = out_size

    @staticmethod
    def _to_bgr(img):
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img

    @staticmethod
    def _to_gray(bgr):
        if len(bgr.shape) == 2:
            return bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def letterbox_resize(img_bgr, out_size=(256, 256), pad_value=127):
        out_w, out_h = out_size
        h, w = img_bgr.shape[:2]
        scale = min(out_w / w, out_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
        x0, y0 = (out_w - nw) // 2, (out_h - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    @staticmethod
    def clahe_bgr(img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def auto_gamma(gray):
        m = np.clip(np.mean(gray) / 255.0, 0.01, 0.99)
        gamma = np.clip(np.log(0.5) / np.log(m), 0.5, 2.0)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(gray, lut)

    @staticmethod
    def detect_green_screen(img_bgr):
        """Detecta si la imagen tiene fondo verde (chroma key)."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Rango de verde en HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(mask > 0) / mask.size
        return green_ratio > 0.2  # Si más del 20% es verde

    @staticmethod
    def segment_green_screen(img_bgr):
        """Segmenta objeto sobre fondo verde."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        # Invertir: objeto = NOT verde
        mask = cv2.bitwise_not(green_mask)
        # Limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def auto_segment(gray, img_bgr=None):
        """
        Segmentación adaptativa universal.
        - Green screen → segmentación por color
        - Fondo oscuro (<50) → objeto es lo claro
        - Fondo claro (>200) → objeto es lo oscuro
        - Ambiguo → minoría es el objeto
        """
        # Si hay imagen color, verificar green screen
        if img_bgr is not None and ImagePreprocessor.detect_green_screen(img_bgr):
            return ImagePreprocessor.segment_green_screen(img_bgr)
        
        # Segmentación por intensidad (Otsu)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detectar tipo de fondo basado en los bordes de la imagen
        h, w = blur.shape
        border = np.concatenate([blur[0,:], blur[-1,:], blur[:,0], blur[:,-1]])
        border_mean = np.mean(border)
        
        if border_mean < 50:
            # Fondo oscuro (Fashion MNIST): objeto debe ser blanco (claro)
            # Después de Otsu, lo claro ya es blanco → no invertir
            pass
        elif border_mean > 200:
            # Fondo claro (Esperma): objeto debe ser blanco
            # Después de Otsu, lo claro es blanco = fondo → invertir
            binary = cv2.bitwise_not(binary)
        else:
            # Ambiguo (padding u otro): usar lógica de minoría
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
        
        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Rellenar contornos externos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        
        return filled

    def run(self, img):
        img_bgr = self._to_bgr(img)
        canon = self.letterbox_resize(img_bgr, self.out_size)
        canon_clahe = self.clahe_bgr(canon)
        gray = self._to_gray(canon_clahe)
        gray = self.auto_gamma(gray)
        mask = self.auto_segment(gray, canon)  # Pasa color para detectar green screen

        return {
            "canon_bgr": canon_clahe,
            "gray": gray,
            "mask": mask
        }


# ============================================================
# CARGA DE DATASET
# ============================================================

def load_zip(zip_path, max_per_class=30, filter_classes=None):
    """
    Carga imágenes de un ZIP.
    filter_classes: lista de clases a incluir, o None para todas
    """
    images, labels = [], []
    with zipfile.ZipFile(zip_path, 'r') as z:
        classes = defaultdict(list)
        for f in z.namelist():
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) or f.endswith('/'):
                continue
            parts = [p for p in f.split('/') if p]  # Quitar vacíos
            if len(parts) >= 2:
                class_name = parts[-2]  # Carpeta padre del archivo
                classes[class_name].append(f)
        
        print(f"\n📂 Todas las clases encontradas: {list(classes.keys())}")
        
        # Filtrar si se especifica
        if filter_classes:
            classes = {k: v for k, v in classes.items() if k in filter_classes}
            print(f"📌 Clases seleccionadas: {list(classes.keys())}")
        
        for cls, files in classes.items():
            count = 0
            for f in files[:max_per_class]:
                data = z.read(f)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    labels.append(cls)
                    count += 1
            print(f"   {cls}: {count} imágenes")
    
    return images, labels


# ============================================================
# VISUALIZACIÓN
# ============================================================

def show(images, labels, prep, n=10):
    indices = np.random.choice(len(images), min(n, len(images)), replace=False)
    
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.5*n))
    
    for row, idx in enumerate(indices):
        v = prep.run(images[idx])
        
        axes[row, 0].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"Original ({labels[idx]})")
        
        axes[row, 1].imshow(cv2.cvtColor(v["canon_bgr"], cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title("Canon + CLAHE")
        
        axes[row, 2].imshow(v["gray"], cmap='gray')
        axes[row, 2].set_title("Gray + Gamma")
        
        axes[row, 3].imshow(v["mask"], cmap='gray')
        axes[row, 3].set_title("Segmentación")
        
        for col in range(4):
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# EJECUTAR
# ============================================================

# Ruta del ZIP ya subido a Colab
zip_path = "/content/originales.zip"  # ← Cambia por tu ruta

# OPCIÓN: Filtrar por clases específicas
# Solo espermatozoides:
# FILTER = ["Abnormal_Sperm", "Non-Sperm", "Normal_Sperm"]

# Solo manos:
# FILTER = ["papel", "piedra", "tijeras"]

# Todas las clases:
FILTER = None

images, labels = load_zip(zip_path, max_per_class=30, filter_classes=FILTER)
print(f"\n✓ Total: {len(images)} imágenes")

prep = ImagePreprocessor(out_size=(256, 256))
show(images, labels, prep, n=12)
