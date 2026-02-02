import cv2
import numpy as np


class ImagePreprocessor:
    """
    Preprocesador universal (sin parámetros del usuario).
    Produce múltiples vistas para soportar diferentes extractores:
      - canon_bgr: embeddings / modelos que usan 3 canales
      - gray: HOG, SIFT/ORB, etc.
      - edges: momentos de forma / HOG robusto / contornos
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
        return img

    @staticmethod
    def _to_gray(bgr):
        if bgr is None:
            return None
        if len(bgr.shape) == 2:
            return bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

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

        scale = min(out_w / w, out_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)

        canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
        x0 = (out_w - nw) // 2
        y0 = (out_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

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
        Gamma automática (sin knobs): estabiliza escenas muy oscuras o muy claras.
        """
        if gray is None:
            return None
        m = float(np.mean(gray)) / 255.0
        m = max(m, 1e-3)
        gamma = np.clip(np.log(0.5) / np.log(m), 0.6, 1.6)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(gray, lut)

    @staticmethod
    def auto_denoise_gray(gray):
        """
        Denoise automático basado en varianza del Laplaciano (estimación simple).
        """
        if gray is None:
            return None
        v = cv2.Laplacian(gray, cv2.CV_64F).var()

        if v > 1200:
            return cv2.GaussianBlur(gray, (5, 5), 0)
        elif v > 400:
            return cv2.medianBlur(gray, 3)
        else:
            return gray

    @staticmethod
    def mild_sharpen(gray):
        """
        Unsharp mask suave: evita kernels agresivos que inventan bordes.
        """
        if gray is None:
            return None
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    # -------------------------
    # Edges automáticos
    # -------------------------

    @staticmethod
    def auto_canny(gray):
        """
        Canny automático con umbrales por mediana.
        """
        if gray is None:
            return None
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        med = np.median(g)
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * med))
        high = int(min(255, (1.0 + sigma) * med))
        return cv2.Canny(g, low, high)

    @staticmethod
    def auto_morph(binary, out_size=(256, 256)):
        """
        Morfología automática basada en tamaño.
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
        """
        img_bgr = self._to_bgr(img)
        if img_bgr is None:
            return None

        # 1) Canonical (forma fija sin distorsión)
        canon = self.letterbox_resize(img_bgr, out_size=self.out_size)

        # 2) Vista para embeddings / estabilidad global
        canon_clahe = self.clahe_on_l_channel(canon)
        gray = self._to_gray(canon_clahe)
        gray = self.auto_denoise_gray(gray)
        gray = self.auto_gamma(gray)
        gray = self.mild_sharpen(gray)

        # 3) Edges
        edges = self.auto_canny(gray)
        edges = self.auto_morph(edges, out_size=self.out_size)

        return {
            "canon_bgr": canon,      # embedding-friendly
            "gray": gray,            # HOG / SIFT / ORB
            "edges": edges           # momentos / contornos
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
