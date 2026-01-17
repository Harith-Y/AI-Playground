"""
Cloudflare R2 Storage Service (S3-compatible)

Handles file upload, download, and deletion using Cloudflare R2.
Falls back to local filesystem if R2 is not configured.
"""
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import Optional, BinaryIO
import logging
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)


class R2StorageService:
    """Service for managing file storage in Cloudflare R2 or local filesystem"""
    
    def __init__(self):
        self.use_r2 = settings.USE_R2_STORAGE and self._validate_r2_config()
        
        if self.use_r2:
            # Initialize R2 client (S3-compatible)
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name='auto'  # R2 uses 'auto' for region
            )
            self.bucket_name = settings.R2_BUCKET_NAME
            logger.info(f"R2 storage initialized with bucket: {self.bucket_name}")
        else:
            self.s3_client = None
            self.local_storage_path = Path(settings.UPLOAD_DIR)
            
            # Try to create directory, fallback to /tmp if permission denied
            try:
                self.local_storage_path.mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = self.local_storage_path / ".test_write"
                test_file.touch()
                test_file.unlink()
                logger.info(f"Local filesystem storage initialized at: {self.local_storage_path}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to initialize storage at {self.local_storage_path}: {e}")
                logger.warning("Falling back to /tmp/uploads")
                self.local_storage_path = Path("/tmp/uploads")
                self.local_storage_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Local filesystem storage initialized at: {self.local_storage_path}")
    
    def _validate_r2_config(self) -> bool:
        """Check if R2 configuration is complete"""
        required = [
            settings.R2_ACCOUNT_ID,
            settings.R2_ACCESS_KEY_ID,
            settings.R2_SECRET_ACCESS_KEY,
            settings.R2_BUCKET_NAME
        ]
        if not all(required):
            logger.warning("R2 configuration incomplete, falling back to local storage")
            return False
        return True
    
    def upload_file(
        self,
        file_content: bytes,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload file to R2 or local storage
        
        Args:
            file_content: File content as bytes
            file_path: Relative path where file should be stored (e.g., "user_123/dataset_456/data.csv")
            content_type: MIME type of the file
        
        Returns:
            str: Public URL or local path to the uploaded file
        """
        if self.use_r2:
            return self._upload_to_r2(file_content, file_path, content_type)
        else:
            return self._upload_to_local(file_content, file_path)
    
    def _upload_to_r2(
        self,
        file_content: bytes,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file to R2 bucket"""
        try:
            # Prepare upload parameters
            upload_params = {
                'Bucket': self.bucket_name,
                'Key': file_path,
                'Body': file_content,
            }
            
            if content_type:
                upload_params['ContentType'] = content_type
            
            # Upload to R2
            self.s3_client.put_object(**upload_params)
            
            # Return public URL if configured, otherwise return S3 URI
            if settings.R2_PUBLIC_URL:
                public_url = f"{settings.R2_PUBLIC_URL}/{file_path}"
            else:
                public_url = f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{self.bucket_name}/{file_path}"
            
            logger.info(f"File uploaded to R2: {file_path}")
            return public_url
            
        except NoCredentialsError:
            logger.error("R2 credentials not available")
            raise ValueError("R2 storage credentials not configured")
        except ClientError as e:
            logger.error(f"Failed to upload to R2: {e}")
            raise ValueError(f"Failed to upload file to R2: {str(e)}")
    
    def _upload_to_local(self, file_content: bytes, file_path: str) -> str:
        """Upload file to local filesystem"""
        try:
            # Create full path
            full_path = self.local_storage_path / file_path
            
            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            full_path.write_bytes(file_content)
            
            logger.info(f"File saved locally: {full_path}")
            return str(full_path)
            
        except Exception as e:
            logger.error(f"Failed to save file locally: {e}")
            raise ValueError(f"Failed to save file: {str(e)}")
    
    def download_file(self, file_path: str) -> bytes:
        """
        Download file from R2 or local storage
        
        Args:
            file_path: Relative path to the file, or full URL for R2 files
        
        Returns:
            bytes: File content
        """
        # If file_path is a URL and we're using R2, extract the key
        if self.use_r2 and (file_path.startswith('http://') or file_path.startswith('https://')):
            file_path = self._extract_r2_key_from_url(file_path)
        
        if self.use_r2:
            return self._download_from_r2(file_path)
        else:
            return self._download_from_local(file_path)
    
    def _extract_r2_key_from_url(self, url: str) -> str:
        """
        Extract R2 key from a URL
        
        Examples:
            https://custom-domain.com/user_id/dataset_id/file.csv -> user_id/dataset_id/file.csv
            https://account.r2.cloudflarestorage.com/bucket/user_id/dataset_id/file.csv -> user_id/dataset_id/file.csv
        """
        try:
            if settings.R2_PUBLIC_URL and url.startswith(settings.R2_PUBLIC_URL):
                # Custom domain URL: remove the domain part
                key = url[len(settings.R2_PUBLIC_URL):].lstrip('/')
                logger.debug(f"Extracted R2 key from custom URL: {key}")
                return key
            else:
                # Standard R2 URL format: https://account.r2.cloudflarestorage.com/bucket/path/to/file
                # Split and take everything after the bucket name
                parts = url.split('/', 5)  # https: / / domain / bucket / path
                if len(parts) > 5:
                    key = parts[5]
                    logger.debug(f"Extracted R2 key from standard URL: {key}")
                    return key
                else:
                    # Fallback: return the URL as-is (might be a relative path)
                    logger.warning(f"Could not parse R2 URL, using as-is: {url}")
                    return url
        except Exception as e:
            logger.error(f"Failed to extract R2 key from URL {url}: {e}")
            return url
    
    def _download_from_r2(self, file_path: str) -> bytes:
        """Download file from R2 bucket"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            content = response['Body'].read()
            logger.info(f"File downloaded from R2: {file_path}")
            return content
            
        except ClientError as e:
            logger.error(f"Failed to download from R2: {e}")
            raise ValueError(f"Failed to download file from R2: {str(e)}")
    
    def _download_from_local(self, file_path: str) -> bytes:
        """Download file from local filesystem"""
        try:
            full_path = self.local_storage_path / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            content = full_path.read_bytes()
            logger.info(f"File read from local storage: {full_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file locally: {e}")
            raise ValueError(f"Failed to read file: {str(e)}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from R2 or local storage
        
        Args:
            file_path: Relative path to the file, or full URL for R2 files
        
        Returns:
            bool: True if successful
        """
        # If file_path is a URL and we're using R2, extract the key
        if self.use_r2 and (file_path.startswith('http://') or file_path.startswith('https://')):
            file_path = self._extract_r2_key_from_url(file_path)
        
        if self.use_r2:
            return self._delete_from_r2(file_path)
        else:
            return self._delete_from_local(file_path)
    
    def _delete_from_r2(self, file_path: str) -> bool:
        """Delete file from R2 bucket"""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            logger.info(f"File deleted from R2: {file_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete from R2: {e}")
            return False
    
    def _delete_from_local(self, file_path: str) -> bool:
        """Delete file from local filesystem"""
        try:
            full_path = self.local_storage_path / file_path
            
            if full_path.exists():
                full_path.unlink()
                logger.info(f"File deleted from local storage: {full_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file locally: {e}")
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes
        
        Args:
            file_path: Either relative path (for local storage) or full URL (for R2)
        
        Returns:
            int: File size in bytes, or 0 if not found
        """
        if self.use_r2:
            # For R2, extract the key from the URL if needed
            if file_path.startswith('http://') or file_path.startswith('https://'):
                # Extract key from URL: https://account.r2.cloudflarestorage.com/bucket/path/to/file
                # or: https://custom-domain.com/path/to/file
                try:
                    if settings.R2_PUBLIC_URL and file_path.startswith(settings.R2_PUBLIC_URL):
                        # Custom domain URL
                        key = file_path[len(settings.R2_PUBLIC_URL):].lstrip('/')
                    else:
                        # Standard R2 URL
                        parts = file_path.split('/', 4)  # Split by first 4 slashes
                        key = parts[4] if len(parts) > 4 else file_path
                    return self._get_file_size_from_r2(key)
                except Exception as e:
                    logger.error(f"Failed to parse R2 URL: {file_path}, error: {e}")
                    return 0
            else:
                return self._get_file_size_from_r2(file_path)
        else:
            return self._get_file_size_locally(file_path)
    
    def _get_file_size_from_r2(self, file_path: str) -> int:
        """Get file size from R2 bucket"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return response.get('ContentLength', 0)
        except ClientError as e:
            logger.error(f"Failed to get file size from R2: {e}")
            return 0
    
    def _get_file_size_locally(self, file_path: str) -> int:
        """Get file size from local filesystem"""
        try:
            full_path = self.local_storage_path / file_path
            if full_path.exists():
                return full_path.stat().st_size
            return 0
        except Exception as e:
            logger.error(f"Failed to get local file size: {e}")
            return 0
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in R2 or local storage
        
        Args:
            file_path: Relative path to the file
        
        Returns:
            bool: True if file exists
        """
        if self.use_r2:
            return self._file_exists_in_r2(file_path)
        else:
            return self._file_exists_locally(file_path)
    
    def _file_exists_in_r2(self, file_path: str) -> bool:
        """Check if file exists in R2 bucket"""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        except ClientError:
            return False
    
    def _file_exists_locally(self, file_path: str) -> bool:
        """Check if file exists in local filesystem"""
        full_path = self.local_storage_path / file_path
        return full_path.exists()
    
    def generate_presigned_url(
        self,
        file_path: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary file access (R2 only)
        
        Args:
            file_path: Relative path to the file
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            str: Presigned URL or None if using local storage
        """
        if not self.use_r2:
            logger.warning("Presigned URLs not available for local storage")
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_path
                },
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL for: {file_path}")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def get_storage_info(self) -> dict:
        """Get current storage configuration info"""
        return {
            "storage_type": "r2" if self.use_r2 else "local",
            "bucket_name": self.bucket_name if self.use_r2 else None,
            "local_path": str(self.local_storage_path) if not self.use_r2 else None,
            "r2_configured": self._validate_r2_config(),
        }


# Singleton instance
_storage_service: Optional[R2StorageService] = None


def get_storage_service() -> R2StorageService:
    """Get or create storage service singleton"""
    global _storage_service
    if _storage_service is None:
        _storage_service = R2StorageService()
    return _storage_service
