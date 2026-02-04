import cv2
import numpy as np


class ImagePreprocessor:
    """
    Preprocesador universal robusto (sin parámetros del usuario).
    Produce múltiples vistas para soportar diferentes extractores:
      - canon_bgr: embeddings / modelos que usan 3 canales
      - gray: HOG, SIFT/ORB, etc.
      - edges: momentos de forma / HOG robusto / contornos

    Cambios v2:
      - Orden corregido: denoise → CLAHE → gamma → sharpen condicional
      - Denoise adaptativo con MAD (más robusto que varianza)
      - Canny con Otsu (más robusto a distribuciones bimodales)
      - Validación de imágenes degeneradas
      - Sharpen condicional (solo si bajo contraste)
    """

    def __init__(self, out_size=(256, 256)):
        self.out_size = out_size  # (width, height)

    # -------------------------
    # Helpers básicos
    # -------------------------

    @staticmethod
    def _to_uint8(img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_bgr(img):
        if img is None:
            return None
        img = ImagePreprocessor._to_uint8(img)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Si es RGBA, convertir a BGR
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img

    @staticmethod
    def _to_gray(bgr):
        if bgr is None:
            return None
        if len(bgr.shape) == 2:
            return bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Validación de imagen
    # -------------------------

    @staticmethod
    def _is_degenerate(gray, min_std=2.0):
        """
        Detecta imágenes degeneradas (uniformes, completamente blancas/negras).
        Retorna True si la imagen no tiene suficiente variación para ser útil.
        """
        if gray is None:
            return True
        std = float(np.std(gray))
        return std < min_std

    @staticmethod
    def _safe_mean(gray):
        """Calcula media normalizada [0,1] con protección contra edge cases."""
        if gray is None:
            return 0.5
        m = float(np.mean(gray)) / 255.0
        # Clamp para evitar problemas con log(0) o log(1)
        return np.clip(m, 0.01, 0.99)

    # -------------------------
    # Canonical resize (sin distorsión)
    # -------------------------

    @staticmethod
    def letterbox_resize(img_bgr, out_size=(256, 256), pad_value=127):
        """
        Resize general SIN distorsión: mantiene aspecto y rellena.
        out_size = (width, height)
        """
        if img_bgr is None:
            return None

        out_w, out_h = out_size
        h, w = img_bgr.shape[:2]

        # Protección contra dimensiones cero
        if h == 0 or w == 0:
            return np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)

        scale = min(out_w / w, out_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        # Asegurar al menos 1 pixel
        nw, nh = max(1, nw), max(1, nh)

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)

        canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
        x0 = (out_w - nw) // 2
        y0 = (out_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # -------------------------
    # Denoise adaptativo (MEJORADO)
    # -------------------------

    @staticmethod
    def auto_denoise_gray(gray):
        """
        Denoise adaptativo usando MAD (Median Absolute Deviation) del Laplaciano.
        MAD es más robusto que la varianza a outliers y texturas estructuradas.
        """
        if gray is None:
            return None

        # Calcular Laplaciano
        lap = cv2.Laplacian(gray, cv2.CV_64F)

        # MAD = median(|x - median(x)|) - más robusto que varianza
        med_lap = np.median(np.abs(lap))

        # Umbrales basados en MAD (escala más estable que varianza)
        # MAD típico de ruido gaussiano: ~1.5 * sigma
        if med_lap > 25:  # Ruido alto
            return cv2.GaussianBlur(gray, (5, 5), 0)
        elif med_lap > 12:  # Ruido moderado
            return cv2.medianBlur(gray, 3)
        else:  # Bajo ruido, preservar detalles
            return gray

    # -------------------------
    # Normalización automática
    # -------------------------

    @staticmethod
    def clahe_on_l_channel(img_bgr, clip=2.0, grid=(8, 8)):
        """
        CLAHE en canal L de LAB: mejora contraste sin destruir colores.
        """
        if img_bgr is None:
            return None
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    @staticmethod
    def auto_gamma(gray):
        """
        Gamma automática robusta: estabiliza escenas muy oscuras o muy claras.
        Usa media clampeada para evitar inestabilidades numéricas.
        """
        if gray is None:
            return None

        m = ImagePreprocessor._safe_mean(gray)

        # gamma = log(0.5) / log(mean) → centra la media en 0.5
        gamma = np.log(0.5) / np.log(m)
        gamma = np.clip(gamma, 0.5, 2.0)  # Rango más amplio pero seguro

        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(gray, lut)

    @staticmethod
    def mild_sharpen(gray, apply_threshold=30.0):
        """
        Unsharp mask suave, SOLO si la imagen tiene bajo contraste local.
        apply_threshold: solo aplicar si std < threshold (imagen borrosa)
        """
        if gray is None:
            return None

        # Solo aplicar sharpen si la imagen parece borrosa
        std = float(np.std(gray))
        if std > apply_threshold:
            return gray  # Ya tiene suficiente contraste

        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    # -------------------------
    # Edges automáticos (MEJORADO)
    # -------------------------

    @staticmethod
    def auto_canny(gray):
        """
        Canny automático usando Otsu para determinar umbral alto.
        Más robusto a distribuciones de intensidad variadas.
        """
        if gray is None:
            return None

        # Suavizado ligero
        g = cv2.GaussianBlur(gray, (3, 3), 0)

        # Otsu para encontrar umbral óptimo
        otsu_thresh, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Usar Otsu como referencia para umbrales de Canny
        high = int(min(255, otsu_thresh * 1.0))
        low = int(max(0, otsu_thresh * 0.5))

        # Asegurar umbrales mínimos para evitar ruido excesivo
        low = max(low, 10)
        high = max(high, low + 10)

        return cv2.Canny(g, low, high)

    @staticmethod
    def auto_morph(binary, out_size=(256, 256)):
        """
        Morfología automática basada en tamaño de salida.
        Cierra pequeños gaps y elimina ruido fino.
        """
        if binary is None:
            return None
        out_w, out_h = out_size
        k = 3 if min(out_w, out_h) >= 128 else 2
        kernel = np.ones((k, k), np.uint8)
        x = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=1)
        return x

    # -------------------------
    # API principal
    # -------------------------

    def run(self, img):
        """
        Ejecuta el preprocesamiento universal y devuelve vistas.

        Pipeline v2 (orden corregido):
        1. Convertir a formato canónico (BGR, tamaño fijo)
        2. Denoise (antes de amplificar ruido con CLAHE)
        3. CLAHE para normalizar contraste
        4. Gamma para balancear luminosidad
        5. Sharpen condicional (solo si bajo contraste)
        6. Edges con Canny+Otsu

        Retorna dict con vistas o None si la imagen es inválida.
        """
        img_bgr = self._to_bgr(img)
        if img_bgr is None:
            return None

        # 1) Canonical (forma fija sin distorsión)
        canon = self.letterbox_resize(img_bgr, out_size=self.out_size)

        # Convertir a gris temprano para validación
        gray_raw = self._to_gray(canon)

        # Validar imagen degenerada
        if self._is_degenerate(gray_raw):
            # Retornar vistas neutras pero válidas (no None)
            # para que el pipeline no falle
            return {
                "canon_bgr": canon,
                "gray": gray_raw,
                "edges": np.zeros(gray_raw.shape, dtype=np.uint8),
                "is_degenerate": True  # Flag para que extractores sepan
            }

        # 2) Denoise PRIMERO (antes de CLAHE para no amplificar ruido)
        gray_denoised = self.auto_denoise_gray(gray_raw)

        # 3) CLAHE en imagen color (para canon_bgr mejorado)
        canon_clahe = self.clahe_on_l_channel(canon)

        # 4) Gamma en escala de grises
        gray = self.auto_gamma(gray_denoised)

        # 5) Sharpen condicional (solo si bajo contraste)
        gray = self.mild_sharpen(gray)

        # 6) Edges con Canny+Otsu
        edges = self.auto_canny(gray)
        edges = self.auto_morph(edges, out_size=self.out_size)

        return {
            "canon_bgr": canon_clahe,  # embedding-friendly, mejorado con CLAHE
            "gray": gray,              # HOG / SIFT / ORB
            "edges": edges,            # momentos / contornos
            "is_degenerate": False
        }


class UniversalPreprocessPipeline:
    """
    Wrapper para que lo uses fácil desde tu backend.
    """

    def __init__(self, out_size=(256, 256)):
        self.pre = ImagePreprocessor(out_size=out_size)

    def preprocess(self, img):
        """
        Retorna vistas para tus extractores.
        """
        views = self.pre.run(img)
        if views is None:
            # consistente: si entra None, sale None
            return None
        return views
