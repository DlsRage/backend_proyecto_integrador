"""
Script de prueba para verificar la conexión con DigitalOcean Spaces (S3)
"""

import sys
import os

# Agregar el directorio raíz al path para importar los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.storage import StorageService
from app.core.config import settings

def test_s3_connection():
    print("🔍 Probando conexión con DigitalOcean Spaces...")
    print(f"   Endpoint: {settings.SPACES_ENDPOINT}")
    print(f"   Bucket: {settings.SPACES_BUCKET}")
    print(f"   Region: {settings.SPACES_REGION}")
    print()

    try:
        # Crear instancia del servicio
        storage = StorageService()
        print("✅ Cliente S3 creado exitosamente")
        
        # Intentar listar objetos en el bucket
        print("\n📋 Listando objetos en el bucket...")
        response = storage.s3.list_objects_v2(
            Bucket=settings.SPACES_BUCKET,
            MaxKeys=10
        )
        
        if 'Contents' in response:
            print(f"✅ Conexión exitosa! Se encontraron {len(response['Contents'])} objetos")
            print("\n📁 Primeros objetos:")
            for obj in response['Contents'][:5]:
                size_kb = obj['Size'] / 1024
                print(f"   - {obj['Key']} ({size_kb:.2f} KB)")
        else:
            print("✅ Conexión exitosa! El bucket está vacío")
        
        # Probar generar URL presignada
        print("\n🔗 Probando generación de URL presignada...")
        test_key = "test/prueba.jpg"
        url = storage.presign_put(test_key, "image/jpeg", expires_sec=60)
        print(f"✅ URL presignada generada exitosamente")
        print(f"   Key: {test_key}")
        print(f"   URL (truncada): {url[:100]}...")
        
        print("\n" + "="*60)
        print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   Mensaje: {str(e)}")
        print("\n🔧 Posibles soluciones:")
        print("   1. Verifica que las credenciales en app/core/config.py sean correctas")
        print("   2. Verifica que el bucket exista y tenga los permisos correctos")
        print("   3. Verifica la conectividad a internet")
        print("   4. Verifica que la región y endpoint sean correctos")
        
        return False

if __name__ == "__main__":
    success = test_s3_connection()
    sys.exit(0 if success else 1)
