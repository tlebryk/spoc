"""Simple Azure Blob Storage interface for saving/loading files."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class AzureStorage:
    """Simple Azure Blob Storage client."""

    def __init__(self, connection_string: Optional[str] = None, container: str = "spoc"):
        """Initialize with connection string from params or AZURE_STORAGE_CONNECTION_STRING env var."""
        if not AZURE_AVAILABLE:
            raise ImportError("Install: pip install azure-storage-blob")

        conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            raise ValueError("Provide connection_string or set AZURE_STORAGE_CONNECTION_STRING")

        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container = self.client.get_container_client(container)

        try:
            self.container.create_container()
        except:
            pass  # Already exists

    def save_json(self, data: Union[Dict, list], path: str):
        """Save JSON to blob."""
        blob = self.container.get_blob_client(path)
        blob.upload_blob(json.dumps(data, indent=2), overwrite=True)
        return blob.url

    def load_json(self, path: str) -> Union[Dict, list]:
        """Load JSON from blob."""
        blob = self.container.get_blob_client(path)
        return json.loads(blob.download_blob().readall())

    def save_file(self, local_path: str, blob_path: Optional[str] = None):
        """Upload file to blob."""
        blob_path = blob_path or Path(local_path).name
        blob = self.container.get_blob_client(blob_path)
        with open(local_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)
        return blob.url

    def load_file(self, blob_path: str, local_path: Optional[str] = None):
        """Download file from blob."""
        local_path = local_path or Path(blob_path).name
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob = self.container.get_blob_client(blob_path)
        with open(local_path, "wb") as f:
            f.write(blob.download_blob().readall())
        return local_path

    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List blobs with optional prefix."""
        return [b.name for b in self.container.list_blobs(name_starts_with=prefix)]

    def exists(self, path: str) -> bool:
        """Check if blob exists."""
        return self.container.get_blob_client(path).exists()
