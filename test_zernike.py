"""
Script de prueba para verificar el funcionamiento de los Momentos de Zernike
"""

import cv2
import numpy as np
from app.services.feature_extractors.moments import MomentsExtractor
from app.services.pipeline import preprocess

def create_test_shapes():
    """Crea formas geométricas de prueba para validar los momentos de Zernike."""
    size = 128
    shapes = {}
    
    # 1. Círculo
    circle = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle, (size//2, size//2), 40, 255, -1)
    shapes['circle'] = circle
    
    # 2. Cuadrado
    square = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(square, (30, 30), (98, 98), 255, -1)
    shapes['square'] = square
    
    # 3. Triángulo
    triangle = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([[64, 20], [20, 108], [108, 108]], np.int32)
    cv2.fillPoly(triangle, [pts], 255)
    shapes['triangle'] = triangle
    
    # 4. Estrella
    star = np.zeros((size, size), dtype=np.uint8)
    center = (size//2, size//2)
    outer_radius = 50
    inner_radius = 20
    pts = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(star, [pts], 255)
    shapes['star'] = star
    
    # 5. Rectángulo horizontal
    rect_h = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(rect_h, (20, 45), (108, 83), 255, -1)
    shapes['rectangle_h'] = rect_h
    
    # 6. Rectángulo vertical
    rect_v = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(rect_v, (45, 20), (83, 108), 255, -1)
    shapes['rectangle_v'] = rect_v
    
    return shapes

def test_zernike_extraction():
    """Prueba la extracción de momentos de Zernike."""
    print("=" * 70)
    print("🧪 PRUEBA DE MOMENTOS DE ZERNIKE")
    print("=" * 70)
    
    # Crear extractor
    extractor = MomentsExtractor(radius=21, degree=8)
    print(f"\n✅ Extractor creado: radius={extractor.radius}, degree={extractor.degree}")
    
    # Crear formas de prueba
    shapes = create_test_shapes()
    print(f"\n✅ Formas de prueba creadas: {list(shapes.keys())}")
    
    # Extraer características de cada forma
    features = {}
    print("\n" + "-" * 70)
    print("📊 EXTRACCIÓN DE CARACTERÍSTICAS")
    print("-" * 70)
    
    for name, shape in shapes.items():
        # Simular el preprocesamiento
        # En producción, usarías preprocess(shape), pero aquí simplificamos
        views = {"edges": shape}
        
        # Extraer momentos de Zernike
        moments = extractor.extract(views)
        features[name] = moments
        
        print(f"\n{name.upper()}:")
        print(f"  - Dimensión del vector: {len(moments)}")
        print(f"  - Rango de valores: [{moments.min():.4f}, {moments.max():.4f}]")
        print(f"  - Norma L2: {np.linalg.norm(moments):.4f}")
        print(f"  - Primeros 5 momentos: {moments[:5]}")
    
    # Calcular simil