import boto3
from botocore.client import Config
from app.core.config import settings

class StorageService:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            region_name=settings.SPACES_REGION,
            endpoint_url=settings.SPACES_ENDPOINT,
            aws_access_key_id=settings.SPACES_KEY,
            aws_secret_access_key=settings.SPACES_SECRET,
            config=Config(signature_version="s3v4"),
        )

    def presign_put(self, key: str, content_type: str, expires_sec: int = 900) -> str:
        return self.s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": settings.SPACES_BUCKET,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_sec,
        )

    def get_object_bytes(self, key: str) -> bytes:
        obj = self.s3.get_object(Bucket=settings.SPACES_BUCKET, Key=key)
        return obj["Body"].read()

    def delete_object(self, key: str) -> None:
        self.s3.delete_object(Bucket=settings.SPACES_BUCKET, Key=key)
