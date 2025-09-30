"""Thin wrapper around the MinIO Python SDK."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from minio import Minio

from src.config import MinIOSettings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class MinioClient:
    client: Minio
    settings: MinIOSettings

    @classmethod
    def from_settings(cls, settings: Optional[MinIOSettings] = None) -> "MinioClient":
        settings = settings or get_settings().minio
        endpoint = settings.endpoint_url.replace("http://", "").replace("https://", "")
        client = Minio(endpoint, access_key=settings.access_key, secret_key=settings.secret_key, secure=settings.secure) # TLS si secured
        return cls(client=client, settings=settings)

    def ensure_bucket(self, bucket: str) -> None:
        if not self.client.bucket_exists(bucket):
            logger.info("Creating missing MinIO bucket: %s", bucket)
            self.client.make_bucket(bucket)

    # Para archivo en disco, leer como bytes
    def upload_file(self, file_path: Path, bucket: Optional[str] = None, object_name: Optional[str] = None) -> str:
        if bucket is None:
            bucket = self.settings.bucket_models
        self.ensure_bucket(bucket)

        if object_name is None:
            object_name = file_path.name

        with file_path.open("rb") as buffer:
            self.client.put_object(bucket, object_name, data=buffer, length=file_path.stat().st_size)
        uri = self.settings.s3_uri(bucket, object_name)
        logger.debug("Uploaded %s to %s", file_path, uri)
        return uri

    # Para conetenido ya en memoria
    def upload_bytes(self, data: bytes, bucket: str, object_name: str, content_type: str = "application/octet-stream") -> str:
        self.ensure_bucket(bucket)
        buffer = io.BytesIO(data)
        self.client.put_object(bucket, object_name, data=buffer, length=len(data), content_type=content_type)
        return self.settings.s3_uri(bucket, object_name)


__all__ = ["MinioClient"]
