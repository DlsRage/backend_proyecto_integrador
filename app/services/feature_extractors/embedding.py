"""
Extractor de características basado en embeddings generados por un modelo CNN
preentrenado. Actualmente utiliza ResNet de torchvision como base y reemplaza
la capa final por una identidad para obtener un vector de características
de tamaño fijo para cada imagen.

La clase sigue la interfaz de ``FeatureExtractor`` definida en ``base.py``
para ser utilizada de manera intercambiable con otros extractores (HOG,
momentos, SIFT, etc.) dentro del pipeline de preprocesamiento y
clustering.

Ejemplo de uso::

    from app.services.feature_extractors.embedding import EmbeddingExtractor

    extractor = EmbeddingExtractor(model_name="resnet50")
    views = preprocess(image_bgr)  # generado por UniversalPreprocessPipeline
    feats = extractor.extract(views)

Los vectores devueltos son de tipo ``numpy.ndarray`` de tipo ``float32``.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from app.services.feature_extractors.base import FeatureExtractor
from PIL import Image
from torchvision import models, transforms


class EmbeddingExtractor(FeatureExtractor):
    """Extractor de embeddings basado en CNN preentrenadas (por defecto ResNet50).

    Este extractor carga un modelo de visión preentrenado (por defecto
    ``resnet50``) de la biblioteca ``torchvision``, reemplazando su capa
    fully-connected final por una identidad para devolver directamente las
    activaciones de la última capa antes de la clasificación. Se puede
    seleccionar el dispositivo de ejecución (``"cuda"`` o ``"cpu"``) y
    el nombre del modelo durante la construcción.
    """

    def __init__(self, model_name: str = "resnet50", device: str | None = None) -> None:
        # Determinar dispositivo: usar GPU si disponible y no se especificó
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Cargar el modelo preentrenado solicitado
        if model_name == "resnet50":
            # Cargar pesos preentrenados; torchvision >= 0.13 usa weights en vez de pretrained
            try:
                weights = models.ResNet50_Weights.DEFAULT  # type: ignore[attr-defined]
                model = models.resnet50(weights=weights)
            except AttributeError:
                # Compatibilidad con versiones más antiguas de torchvision
                model = models.resnet50(pretrained=True)
            # Reemplazar la capa fully-connected por identidad
            output_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif model_name == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT  # type: ignore[attr-defined]
                model = models.resnet18(weights=weights)
            except AttributeError:
                model = models.resnet18(pretrained=True)
            output_dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            raise ValueError(
                f"Modelo {model_name!r} no soportado para extracción de embeddings"
            )

        model.eval()
        model.to(self.device)

        self.model = model
        self.output_dim = output_dim

        # Definir transformaciones de entrada: redimensionar, tensorizar y normalizar
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, views: dict[str, np.ndarray]) -> np.ndarray:
        """
        Extrae un vector de características a partir de las vistas preprocesadas.

        Este extractor espera que el diccionario ``views`` contenga la clave
        ``"canon_bgr"``, que corresponde a una imagen en formato BGR (según
        OpenCV) ya preprocesada por ``UniversalPreprocessPipeline``.

        Args:
            views: Diccionario de vistas que incluye ``canon_bgr``.

        Returns:
            Un vector ``numpy.ndarray`` de dimensiones (``output_dim``,) con
            el embedding generado por la CNN.
        """
        bgr = views.get("canon_bgr")
        if bgr is None:
            raise KeyError(
                "La vista 'canon_bgr' no está presente en las vistas proporcionadas"
            )

        # Convertir de BGR (OpenCV) a RGB para PIL
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Aplicar transformaciones y agregar dimensión de batch
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Ejecutar modelo en modo no gradiente
        with torch.no_grad():
            embedding = self.model(tensor)

        # Convertir a numpy, aplanar y forzar tipo float32
        vec = embedding.cpu().numpy().reshape(-1).astype(np.float32)
        return vec
