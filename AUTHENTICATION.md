# Authentication & Authorization Implementation Guide

This document describes the JWT-based authentication and authorization system implemented for the AI Playground API.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Security Features](#security-features)
- [API Endpoints](#api-endpoints)
- [Usage Guide](#usage-guide)
- [Migration Guide](#migration-guide)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The AI Playground now implements comprehensive JWT (JSON Web Token) based authentication with role-based access control (RBAC). All sensitive endpoints are protected and require valid authentication.

### Key Features

✅ **JWT Authentication** - Secure token-based authentication
✅ **Password Hashing** - Bcrypt password hashing
✅ **Role-Based Access Control** - Admin and regular user roles
✅ **Resource Ownership Verification** - Users can only access their own resources
✅ **API Key Support** - Optional API keys for programmatic access
✅ **Refresh Tokens** - Long-lived refresh tokens for seamless re-authentication
✅ **Password Requirements** - Strong password validation

## Architecture

### Components

```
backend/app/
├── core/
│   ├── security.py          # JWT & auth utilities
│   └── config.py            # Security settings
├── models/
│   └── user.py              # User model with auth fields
├── schemas/
│   └── user.py              # Auth request/response schemas
└── api/v1/endpoints/
    └── auth.py              # Authentication endpoints
```

### Authentication Flow

```
1. User Registration
   POST /api/v1/auth/register
   → Create user with hashed password
   → Return user profile

2. User Login
   POST /api/v1/auth/login
   → Verify credentials
   → Generate access & refresh tokens
   → Return tokens

3. Protected Endpoint Access
   GET /api/v1/datasets
   Header: Authorization: Bearer {access_token}
   → Validate JWT token
   → Extract user ID
   → Process request

4. Token Refresh
   POST /api/v1/auth/refresh
   → Validate refresh token
   → Generate new access & refresh tokens
   → Return new tokens
```

## Security Features

### Password Security

**Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit

**Hashing:**
- Algorithm: Bcrypt
- Automatically salted
- Computationally expensive (prevents brute-force)

### JWT Tokens

**Access Token:**
- Short-lived (default: 30 minutes)
- Used for API authentication
- Contains user ID in `sub` claim

**Refresh Token:**
- Long-lived (30 days)
- Used to obtain new access tokens
- Marked with `type: refresh` claim

**Configuration:**
```env
SECRET_KEY=your-secret-key-minimum-64-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Authorization Levels

| Level | Description | Capabilities |
|-------|-------------|--------------|
| **Regular User** | Default role | Access own resources only |
| **Admin** | Elevated privileges | Access all resources, admin operations |
| **Inactive** | Deactivated account | Cannot login or access resources |

### Resource Ownership

All resources (datasets, experiments, models) include ownership verification:

```python
verify_resource_ownership(
    resource_user_id=dataset.user_id,
    current_user_id=user.id,
    allow_admin=True,  # Admins can access any resource
    db=db
)
```

## API Endpoints

### Authentication Endpoints

All auth endpoints are under `/api/v1/auth`:

#### Register New User

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "is_active": true,
  "is_admin": false,
  "created_at": "2026-01-01T10:00:00Z",
  "updated_at": "2026-01-01T10:00:00Z",
  "last_login": null
}
```

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Refresh Access Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Get Current User Profile

```http
GET /api/v1/auth/me
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "is_active": true,
  "is_admin": false,
  "created_at": "2026-01-01T10:00:00Z",
  "updated_at": "2026-01-01T10:00:00Z",
  "last_login": "2026-01-01T11:30:00Z"
}
```

#### Change Password

```http
POST /api/v1/auth/change-password
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "current_password": "SecurePass123",
  "new_password": "NewSecurePass456"
}
```

**Response:**
```json
{
  "message": "Password changed successfully"
}
```

#### Generate API Key

```http
POST /api/v1/auth/api-key/generate
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "api_key": "aip_ZXlKaGJHY2lPaUpJVXpJMU5pSXNJblI1Y0NJNklrcFhWQ0o5",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "created_at": "2026-01-01T12:00:00Z"
}
```

#### Revoke API Key

```http
DELETE /api/v1/auth/api-key/revoke
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "API key revoked successfully"
}
```

#### Logout

```http
POST /api/v1/auth/logout
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "Logged out successfully. Please delete your tokens."
}
```

*Note: Since JWTs are stateless, actual logout happens client-side by deleting tokens.*

### Protected Endpoints

All other API endpoints now require authentication. Include the JWT token in the `Authorization` header:

```http
GET /api/v1/datasets
Authorization: Bearer {access_token}
```

## Usage Guide

### Frontend Integration

#### 1. Store Tokens Securely

```javascript
// After login
const response = await fetch('/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password })
});

const { access_token, refresh_token } = await response.json();

// Store in memory or secure storage (not localStorage for production)
sessionStorage.setItem('access_token', access_token);
sessionStorage.setItem('refresh_token', refresh_token);
```

#### 2. Include Token in Requests

```javascript
const response = await fetch('/api/v1/datasets', {
  headers: {
    'Authorization': `Bearer ${sessionStorage.getItem('access_token')}`
  }
});
```

#### 3. Handle Token Expiration

```javascript
async function fetchWithAuth(url, options = {}) {
  const access_token = sessionStorage.getItem('access_token');

  let response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${access_token}`
    }
  });

  // If unauthorized, try refreshing token
  if (response.status === 401) {
    const refresh_token = sessionStorage.getItem('refresh_token');

    const refreshResponse = await fetch('/api/v1/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token })
    });

    if (refreshResponse.ok) {
      const { access_token: newAccessToken, refresh_token: newRefreshToken } =
        await refreshResponse.json();

      sessionStorage.setItem('access_token', newAccessToken);
      sessionStorage.setItem('refresh_token', newRefreshToken);

      // Retry original request
      response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${newAccessToken}`
        }
      });
    } else {
      // Refresh failed, redirect to login
      window.location.href = '/login';
    }
  }

  return response;
}
```

### cURL Examples

#### Register
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123"}'
```

#### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123"}'
```

#### Access Protected Endpoint
```bash
curl -X GET http://localhost:8000/api/v1/datasets \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Python Client Example

```python
import requests

class AIPlaygroundClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None

    def register(self, email, password):
        response = requests.post(
            f"{self.base_url}/api/v1/auth/register",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        return response.json()

    def login(self, email, password):
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        return data

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_datasets(self):
        response = requests.get(
            f"{self.base_url}/api/v1/datasets",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def upload_dataset(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/api/v1/datasets/upload",
                headers=self._get_headers(),
                files=files
            )
        response.raise_for_status()
        return response.json()

# Usage
client = AIPlaygroundClient()
client.register("user@example.com", "SecurePass123")
client.login("user@example.com", "SecurePass123")
datasets = client.get_datasets()
```

## Migration Guide

### Running the Database Migration

```bash
cd backend

# Review the migration
alembic history

# Run the migration
alembic upgrade head

# Verify migration
alembic current
```

### Creating First Admin User

After running migrations, existing users will have placeholder password hashes. To create an admin user:

```python
# Python script or Django shell
from app.db.session import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash
import uuid

db = SessionLocal()

# Create admin user
admin = User(
    id=uuid.uuid4(),
    email="admin@example.com",
    password_hash=get_password_hash("AdminSecurePass123"),
    is_active=True,
    is_admin=True
)

db.add(admin)
db.commit()
db.close()
```

### Updating Existing Users

Existing users will need to register again or have passwords reset manually:

```python
from app.core.security import get_password_hash

# Reset user password
user = db.query(User).filter(User.email == "user@example.com").first()
user.password_hash = get_password_hash("NewSecurePass123")
db.commit()
```

## Security Best Practices

### Production Configuration

**1. Strong SECRET_KEY**

Generate a secure secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Update `.env`:
```env
SECRET_KEY=your-generated-secret-key-min-64-characters
```

**2. HTTPS Only**

Always use HTTPS in production. Configure your reverse proxy (nginx/Apache) for SSL/TLS.

**3. Secure Cookie Settings**

If using cookies for tokens (not recommended for APIs):
```python
secure=True  # HTTPS only
httponly=True  # Prevents JavaScript access
samesite='strict'  # CSRF protection
```

**4. Rate Limiting**

Implement rate limiting on auth endpoints to prevent brute-force attacks:

```python
# In main.py or middleware
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")
async def login(...):
    ...
```

**5. Token Expiration**

- Keep access tokens short-lived (15-30 minutes)
- Use refresh tokens for extended sessions
- Consider implementing token blacklisting for logout

**6. Password Policy**

The current implementation enforces:
- Minimum 8 characters
- Mixed case
- At least one digit

Consider adding:
- Special characters requirement
- Password history (prevent reuse)
- Account lockout after failed attempts

**7. Security Headers**

Add security headers in main.py:

```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

## Troubleshooting

### Common Issues

#### 401 Unauthorized

**Symptom:** `{"detail": "Could not validate credentials"}`

**Causes:**
- Expired access token → Use refresh endpoint
- Invalid token format → Check Authorization header format
- Wrong SECRET_KEY → Verify environment configuration

#### 403 Forbidden

**Symptom:** `{"detail": "You don't have permission to access this resource"}`

**Causes:**
- Accessing another user's resource
- Inactive user account
- Non-admin accessing admin endpoint

#### Password Validation Failed

**Symptom:** `{"detail": "Password must contain at least one uppercase letter"}`

**Solution:** Ensure password meets all requirements:
- ≥8 characters
- Upper + lowercase
- At least one digit

#### Database Migration Failed

**Symptom:** Migration errors during `alembic upgrade`

**Solution:**
```bash
# Check current version
alembic current

# Review migration history
alembic history

# Downgrade if needed
alembic downgrade -1

# Try upgrade again
alembic upgrade head
```

### Debugging Tips

#### Enable Debug Logging

```env
LOG_LEVEL=DEBUG
```

#### Test JWT Token

```python
from jose import jwt
from app.core.config import settings

token = "your-token-here"
payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
print(payload)
```

#### Verify Password Hash

```python
from app.core.security import verify_password

is_valid = verify_password("SecurePass123", user.password_hash)
print(f"Password valid: {is_valid}")
```

## Additional Resources

- [JWT.io](https://jwt.io/) - JWT debugger and documentation
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python-JOSE Documentation](https://python-jose.readthedocs.io/)
- [Passlib Documentation](https://passlib.readthedocs.io/)

## Support

For issues or questions:
1. Check this documentation
2. Review API logs: `docker-compose logs backend`
3. Test with cURL to isolate frontend/backend issues
4. Open an issue on GitHub with detailed error messages
