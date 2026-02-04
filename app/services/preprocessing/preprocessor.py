"""
Preprocesador Universal v3
- Letterbox resize
- CLAHE
- Gamma automático
- Segmentación adaptativa (green screen + Otsu)
"""

import cv2
import numpy as np


class UniversalPreprocessPipeline:
    """
    Preprocesador universal para clustering de imágenes.
    Genera vistas múltiples: canon_bgr, gray, mask
    """
    
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
        """Redimensiona manteniendo proporción, con padding."""
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
        """Aplica CLAHE en canal L (LAB)."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def auto_gamma(gray):
        """Corrección gamma adaptativa."""
        m = np.clip(np.mean(gray) / 255.0, 0.01, 0.99)
        gamma = np.clip(np.log(0.5) / np.log(m), 0.5, 2.0)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(gray, lut)

    @staticmethod
    def detect_green_screen(img_bgr):
        """Detecta si la imagen tiene fondo verde (chroma key)."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(mask > 0) / mask.size
        return green_ratio > 0.2

    @staticmethod
    def segment_green_screen(img_bgr):
        """Segmenta objeto sobre fondo verde."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(green_mask)
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
        # Verificar green screen
        if img_bgr is not None and UniversalPreprocessPipeline.detect_green_screen(img_bgr):
            return UniversalPreprocessPipeline.segment_green_screen(img_bgr)
        
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
        
        # Rellenar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        
        return filled

    def preprocess(self, img):
        """
        Preprocesa imagen y retorna vistas.
        
        Returns:
            dict con keys: canon_bgr, gray, mask
        """
        img_bgr = self._to_bgr(img)
        if img_bgr is None:
            return None
            
        canon = self.letterbox_resize(img_bgr, self.out_size)
        canon_clahe = self.clahe_bgr(canon)
        gray = self._to_gray(canon_clahe)
        gray = self.auto_gamma(gray)
        mask = self.auto_segment(gray, canon)

        return {
            "canon_bgr": canon_clahe,
            "gray": gray,
            "mask": mask
        }


# Alias para compatibilidad
ImagePreprocessor = UniversalPreprocessPipeline
