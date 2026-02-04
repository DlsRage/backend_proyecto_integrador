# ============================================================
# TEST COMPLETO DEL PIPELINE EN GOOGLE COLAB
# Preprocesamiento + Extractores de características + Clustering
# Copy-paste este código completo en Colab
# ============================================================

# %% 
# !pip install opencv-python-headless numpy matplotlib scikit-image scikit-learn

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import urllib.request

# ============================================================
# 1. PREPROCESADOR (v2 - Robusto)
# ============================================================

class ImagePreprocessor:
    """
    Preprocesador universal robusto.
    Produce vistas: canon_bgr, gray, edges
    """

    def __init__(self, out_size=(256, 256)):
        self.out_size = out_size

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

    @staticmethod
    def _is_degenerate(gray, min_std=2.0):
        if gray is None:
            return True
        return float(np.std(gray)) < min_std

    @staticmethod
    def _safe_mean(gray):
        if gray is None:
            return 0.5
        m = float(np.mean(gray)) / 255.0
        return np.clip(m, 0.01, 0.99)

    @staticmethod
    def letterbox_resize(img_bgr, out_size=(256, 256), pad_value=127):
        if img_bgr is None:
            return None
        out_w, out_h = out_size
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
        scale = min(out_w / w, out_h / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
        canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
        x0, y0 = (out_w - nw) // 2, (out_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    @staticmethod
    def auto_denoise_gray(gray):
        if gray is None:
            return None
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        med_lap = np.median(np.abs(lap))
        if med_lap > 25:
            return cv2.GaussianBlur(gray, (5, 5), 0)
        elif med_lap > 12:
            return cv2.medianBlur(gray, 3)
        return gray

    @staticmethod
    def clahe_on_l_channel(img_bgr, clip=2.0, grid=(8, 8)):
        if img_bgr is None:
            return None
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        L2 = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def auto_gamma(gray):
        if gray is None:
            return None
        m = ImagePreprocessor._safe_mean(gray)
        gamma = np.clip(np.log(0.5) / np.log(m), 0.5, 2.0)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(gray, lut)

    @staticmethod
    def mild_sharpen(gray, apply_threshold=30.0):
        if gray is None:
            return None
        if float(np.std(gray)) > apply_threshold:
            return gray
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    @staticmethod
    def auto_canny(gray):
        if gray is None:
            return None
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        otsu_thresh, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = int(min(255, otsu_thresh))
        low = max(10, int(otsu_thresh * 0.5))
        high = max(high, low + 10)
        return cv2.Canny(g, low, high)

    @staticmethod
    def auto_morph(binary, out_size=(256, 256)):
        if binary is None:
            return None
        k = 3 if min(out_size) >= 128 else 2
        kernel = np.ones((k, k), np.uint8)
        x = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=1)

    def run(self, img):
        img_bgr = self._to_bgr(img)
        if img_bgr is None:
            return None

        canon = self.letterbox_resize(img_bgr, out_size=self.out_size)
        gray_raw = self._to_gray(canon)

        if self._is_degenerate(gray_raw):
            return {
                "canon_bgr": canon,
                "gray": gray_raw,
                "edges": np.zeros(gray_raw.shape, dtype=np.uint8),
                "is_degenerate": True
            }

        gray_denoised = self.auto_denoise_gray(gray_raw)
        canon_clahe = self.clahe_on_l_channel(canon)
        gray = self.auto_gamma(gray_denoised)
        gray = self.mild_sharpen(gray)
        edges = self.auto_canny(gray)
        edges = self.auto_morph(edges, out_size=self.out_size)

        return {
            "canon_bgr": canon_clahe,
            "gray": gray,
            "edges": edges,
            "is_degenerate": False
        }


# ============================================================
# 2. EXTRACTORES DE CARACTERÍSTICAS (usan views del preprocesador)
# ============================================================

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, views: dict) -> np.ndarray:
        """Recibe dict de vistas, retorna vector 1D."""
        pass


class HOGExtractor(FeatureExtractor):
    def extract(self, views: dict) -> np.ndarray:
        from skimage.feature import hog
        gray = views["gray"]
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return feat.astype(np.float32)


class MomentsExtractor(FeatureExtractor):
    def extract(self, views: dict) -> np.ndarray:
        # Usa edges para momentos de forma
        edges = views["edges"]
        m = cv2.moments(edges)
        keys = ["m00","m10","m01","m20","m11","m02","m30","m21","m12","m03",
                "mu20","mu11","mu02","mu30","mu21","mu12","mu03",
                "nu20","nu11","nu02","nu30","nu21","nu12","nu03"]
        return np.array([m[k] for k in keys], dtype=np.float32)


class SIFTExtractor(FeatureExtractor):
    def __init__(self, max_kp: int = 32):
        self.sift = cv2.SIFT_create()
        self.max_kp = max_kp

    def extract(self, views: dict) -> np.ndarray:
        gray = views["gray"]
        kps, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None or len(desc) == 0:
            return np.zeros((self.max_kp * 128,), dtype=np.float32)
        
        desc = desc[:self.max_kp]
        if desc.shape[0] < self.max_kp:
            pad = np.zeros((self.max_kp - desc.shape[0], 128), dtype=desc.dtype)
            desc = np.vstack([desc, pad])
        
        return desc.reshape(-1).astype(np.float32)


# ============================================================
# 3. PIPELINE COMPLETO
# ============================================================

class ImagePipeline:
    """Pipeline completo: preprocesamiento → extracción de características."""
    
    def __init__(self, extractor: FeatureExtractor, out_size=(256, 256)):
        self.preprocessor = ImagePreprocessor(out_size=out_size)
        self.extractor = extractor
    
    def process(self, img_bgr: np.ndarray) -> dict:
        """
        Procesa una imagen y retorna vistas + features.
        """
        views = self.preprocessor.run(img_bgr)
        if views is None:
            return None
        
        features = self.extractor.extract(views)
        
        return {
            "views": views,
            "features": features,
            "feature_dim": len(features),
            "is_degenerate": views["is_degenerate"]
        }


# ============================================================
# 4. CLUSTERING ONLINE (simplificado para demo)
# ============================================================

class OnlineKMeans:
    """K-Means online simplificado para demo."""
    
    def __init__(self, k=3, lr=0.1):
        self.k = k
        self.lr = lr
        self.centroids = None
        self.counts = None
    
    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
            self.counts = np.array([1])
            return 0
        
        if len(self.centroids) < self.k:
            self.centroids = np.vstack([self.centroids, x])
            self.counts = np.append(self.counts, 1)
            return len(self.centroids) - 1
        
        dists = np.linalg.norm(self.centroids - x, axis=1)
        j = np.argmin(dists)
        
        eta = self.lr / (1 + np.sqrt(self.counts[j]))
        self.centroids[j] += eta * (x - self.centroids[j])
        self.counts[j] += 1
        
        return j


# ============================================================
# 5. FUNCIONES DE TEST
# ============================================================

def load_image_from_url(url):
    resp = urllib.request.urlopen(url)
    img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def test_full_pipeline(images, extractor_name="HOG"):
    """
    Prueba el pipeline completo con una lista de imágenes.
    """
    # Seleccionar extractor
    extractors = {
        "HOG": HOGExtractor(),
        "Moments": MomentsExtractor(),
        "SIFT": SIFTExtractor(max_kp=16)
    }
    extractor = extractors.get(extractor_name, HOGExtractor())
    
    # Crear pipeline
    pipeline = ImagePipeline(extractor=extractor, out_size=(256, 256))
    
    # Crear clustering
    clustering = OnlineKMeans(k=3, lr=0.1)
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"🧪 TEST PIPELINE COMPLETO - Extractor: {extractor_name}")
    print(f"{'='*60}")
    
    for i, img in enumerate(images):
        result = pipeline.process(img)
        
        if result is None:
            print(f"❌ Imagen {i}: No se pudo procesar")
            continue
        
        # Clustering
        cluster_id = clustering.partial_fit(result["features"])
        result["cluster"] = cluster_id
        results.append(result)
        
        status = "⚠️ DEGENERADA" if result["is_degenerate"] else "✓ OK"
        print(f"📷 Imagen {i}: {status} | Features: {result['feature_dim']}D | Cluster: {cluster_id}")
    
    return results, clustering


def visualize_results(images, results):
    """Visualiza imágenes coloreadas por cluster."""
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (img, res) in enumerate(zip(images, results)):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        
        # Mostrar imagen preprocesada
        ax.imshow(cv2.cvtColor(res["views"]["canon_bgr"], cv2.COLOR_BGR2RGB))
        
        cluster = res.get("cluster", 0)
        color = colors[cluster % len(colors)]
        ax.set_title(f"Cluster {cluster}", color=color, fontweight='bold')
        
        # Borde de color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Ocultar ejes vacíos
    for i in range(len(results), rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')
    
    plt.suptitle("Resultados de Clustering", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. EJECUTAR TESTS
# ============================================================

print("=" * 60)
print("🚀 PIPELINE DE VISIÓN POR COMPUTADOR - TEST COMPLETO")
print("=" * 60)

# Generar imágenes de prueba variadas
test_images = []

# Imágenes sintéticas con diferentes características
np.random.seed(42)

# Cluster 0: Imágenes oscuras con bordes verticales
for _ in range(3):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, 50:60] = [80, 80, 80]
    img[:, 140:150] = [80, 80, 80]
    img += np.random.randint(0, 20, img.shape, dtype=np.uint8)
    test_images.append(img)

# Cluster 1: Imágenes claras con círculos
for _ in range(3):
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (150, 150, 200), -1)
    img += np.random.randint(0, 30, img.shape, dtype=np.uint8)
    test_images.append(img)

# Cluster 2: Imágenes con gradiente diagonal
for _ in range(3):
    x = np.linspace(0, 255, 200)
    y = np.linspace(0, 255, 200)
    xx, yy = np.meshgrid(x, y)
    img = ((xx + yy) / 2).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    test_images.append(img)

# Test con diferentes extractores
for extractor_name in ["HOG", "Moments", "SIFT"]:
    results, clustering = test_full_pipeline(test_images, extractor_name)
    
    print(f"\n📊 Distribución de clusters ({extractor_name}):")
    for j in range(clustering.k):
        count = sum(1 for r in results if r.get("cluster") == j)
        print(f"   Cluster {j}: {count} imágenes")

# Visualizar último resultado
print("\n🎨 Visualizando resultados...")
visualize_results(test_images, results)

print("\n" + "=" * 60)
print("✅ TEST COMPLETO FINALIZADO")
print("=" * 60)

# %%
# ============================================================
# PRUEBA CON TUS PROPIAS IMÁGENES
# ============================================================

# from google.colab import files
# uploaded = files.upload()
# 
# my_images = []
# for filename in uploaded.keys():
#     img_bytes = uploaded[filename]
#     img_array = np.frombuffer(img_bytes, dtype=np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     my_images.append(img)
# 
# results, clustering = test_full_pipeline(my_images, "HOG")
# visualize_results(my_images, results)
