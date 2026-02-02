"""
Script para debuggear el procesamiento de jobs
"""
import asyncio
import sys
from app.services.job_manager import JobManager
from app.services.storage import StorageService

async def test_job():
    print("🔧 Iniciando debug del job...")
    
    # Crear job manager
    jm = JobManager()
    
    # Crear un job de prueba
    job = jm.create_job(extractor="hog", n_clusters=3, auto_delete=False)
    print(f"✅ Job creado: {job.id}")
    
    # Registrar algunas imágenes de ejemplo (keys que deberían existir en S3)
    test_keys = [
        "espermatozoides/Normal_Sperm/Normal_Sperm (119).bmp",
        "espermatozoides/Normal_Sperm/Normal_Sperm (179).bmp",
        "espermatozoides/Non-Sperm/Non-Sperm (107).bmp",
    ]
    
    print(f"\n📝 Registrando {len(test_keys)} imágenes...")
    jm.register_images(job.id, test_keys)
    print(f"✅ Imágenes registradas: {job.image_keys}")
    
    # Intentar descargar una imagen de S3 para verificar conexión
    print(f"\n📥 Probando descarga desde S3...")
    storage = StorageService()
    try:
        img_bytes = storage.get_object_bytes(test_keys[0])
        print(f"✅ Descarga exitosa: {len(img_bytes)} bytes")
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return
    
    # Iniciar el job
    print(f"\n🚀 Iniciando procesamiento...")
    await jm.start(job.id)
    
    # Esperar un poco para que procese
    for i in range(10):
        await asyncio.sleep(2)
        print(f"⏳ Estado: {job.status}")
        if job.status in ("done", "failed"):
            break
    
    print(f"\n📊 Resultado final:")
    print(f"  Estado: {job.status}")
    if job.result:
        print(f"  Result: {job.result}")
    else:
        print(f"  No hay resultado disponible")

if __name__ == "__main__":
    asyncio.run(test_job())
