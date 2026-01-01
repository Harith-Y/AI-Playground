# Authentication Implementation Guide for Remaining Endpoints

This guide shows how to add authentication to the remaining endpoint files that still use mock authentication.

## Quick Reference

### Import Required Auth Dependencies

Add to the top of endpoint files:

```python
from uuid import UUID
from app.core.security import get_current_user_id, verify_resource_ownership, get_current_admin_user
from app.db.session import get_db
```

### Replace Mock Function

**Remove:**
```python
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"
```

**Use:**
```python
from app.core.security import get_current_user_id
```

### Update Function Signatures

**Before:**
```python
async def my_endpoint(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
```

**After:**
```python
async def my_endpoint(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
```

### Add Ownership Verification

For endpoints accessing specific resources (GET/PUT/DELETE /{id}):

**Pattern:**
```python
# 1. Get the resource (without filtering by user_id)
resource = db.query(Model).filter(Model.id == resource_id).first()

if not resource:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Resource {resource_id} not found"
    )

# 2. Verify ownership (allows admins to access any resource)
verify_resource_ownership(resource.user_id, user_id, allow_admin=True, db=db)

# 3. Process the resource
return resource
```

## Endpoints to Update

### 1. models.py

**File:** `backend/app/api/v1/endpoints/models.py`

**Location:** Line 48

**Current Mock:**
```python
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"
```

**Changes Required:**

1. Remove mock function
2. Import: `from app.core.security import get_current_user_id, verify_resource_ownership, get_current_admin_user`
3. Update all function parameters: `user_id: str` → `user_id: UUID`
4. Add ownership checks for:
   - `get_training_status(model_run_id, user_id)` - line ~XXX
   - `get_training_result(model_run_id, user_id)` - line ~XXX
   - `delete_model_run(model_run_id, user_id)` - line ~XXX
   - `get_model_metrics(model_run_id, user_id)` - line ~XXX
   - `get_feature_importance(model_run_id, user_id)` - line ~XXX

5. Protect admin endpoint:
   - `clear_cache()` - Use `Depends(get_current_admin_user)` instead of `get_current_user_id`

**Example:**
```python
@router.get("/train/{model_run_id}/result")
async def get_training_result(
    model_run_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    # Get model run
    model_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()

    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run {model_run_id} not found"
        )

    # Verify ownership (get associated experiment to check user_id)
    experiment = db.query(Experiment).filter(Experiment.id == model_run.experiment_id).first()
    if experiment:
        verify_resource_ownership(experiment.user_id, user_id, allow_admin=True, db=db)

    return model_run
```

### 2. preprocessing.py

**File:** `backend/app/api/v1/endpoints/preprocessing.py`

**Location:** Line 49

**Current Mock:**
```python
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"
```

**Changes Required:**

1. Remove mock function
2. Import auth dependencies
3. Update all `user_id: str` → `user_id: UUID`
4. Add ownership checks for preprocessing steps
5. Note: `/task/{task_id}` endpoint (line ~XXX) has **NO AUTH** - add `user_id: UUID = Depends(get_current_user_id)`

**Critical Security Issue:**
```python
@router.get("/task/{task_id}")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    # THIS ENDPOINT HAS NO AUTHENTICATION!
    # Anyone can check any task status
```

**Fix:**
```python
@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)  # ADD THIS
):
    # Verify task belongs to user (if you store user_id with tasks)
    # Or verify the associated resource belongs to user
```

### 3. tuning.py

**File:** `backend/app/api/v1/endpoints/tuning.py`

**Location:** Line 35

**Current Mock:**
```python
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"
```

**Changes Required:**

1. Remove mock function
2. Import auth dependencies
3. Update all `user_id: str` → `user_id: UUID`
4. Add ownership checks for tuning results

### 4. tuning_orchestration.py

**File:** `backend/app/api/v1/endpoints/tuning_orchestration.py`

**Location:** Line 37

**Current Mock:**
```python
def get_current_user_id() -> str:
    """Mock function to get current user ID. Replace with actual auth."""
    return "00000000-0000-0000-0000-000000000001"
```

**Changes Required:**

1. Remove mock function
2. Import auth dependencies
3. Update all `user_id: str` → `user_id: UUID`

### 5. experiment_config.py

**File:** `backend/app/api/v1/endpoints/experiment_config.py`

**Location:** Unknown (check file)

**CRITICAL SECURITY ISSUE:**

Some endpoints in this file have **NO AUTHENTICATION**:

```python
@router.get("/{experiment_id}/config")
async def get_experiment_config(experiment_id: str, db: Session = Depends(get_db)):
    # NO AUTH - anyone can download any experiment config!
```

**Fix Required:**

```python
@router.get("/{experiment_id}/config")
async def get_experiment_config(
    experiment_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)  # ADD AUTH
):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # ADD OWNERSHIP VERIFICATION
    verify_resource_ownership(experiment.user_id, user_id, allow_admin=True, db=db)

    # Return config...
```

### 6. code_generation.py

**File:** `backend/app/api/v1/endpoints/code_generation.py`

**Current Status:** Likely has mock auth

**Changes Required:**

1. Remove mock function
2. Import auth dependencies
3. Add ownership verification for experiments before generating code
4. Code generation is sensitive - ensure proper authorization

### 7. pipelines.py

**File:** `backend/app/api/v1/endpoints/pipelines.py`

**Current Status:** Unknown

**Changes Required:**

1. Check for mock auth
2. Add proper authentication
3. Add ownership verification for pipelines

### 8. features.py

**File:** `backend/app/api/v1/endpoints/features.py`

**Current Status:** Unknown

**Changes Required:**

1. Check for mock auth
2. Add proper authentication if missing

## Admin-Only Endpoints

For operations that should only be accessible to administrators:

### Pattern

```python
from app.core.security import get_current_admin_user
from app.models.user import User

@router.delete("/cache/clear")
async def clear_cache(
    current_user: User = Depends(get_current_admin_user)  # Requires admin
):
    # Only admins can clear cache
    cache.clear()
    return {"message": "Cache cleared"}
```

### Admin Operations

Identify and protect these operations:

- Cache management (`/models/cache/clear`)
- System metrics (if sensitive)
- Bulk operations
- User management (future)
- System configuration changes

## Testing Checklist

After updating each endpoint file:

### 1. Verify Imports

```python
from uuid import UUID
from app.core.security import (
    get_current_user_id,
    verify_resource_ownership,
    get_current_admin_user  # If needed
)
```

### 2. Check Function Signatures

All functions should have:
```python
user_id: UUID = Depends(get_current_user_id)
```

### 3. Verify Ownership Checks

For resource-specific endpoints:
```python
verify_resource_ownership(resource.user_id, user_id, allow_admin=True, db=db)
```

### 4. Test with cURL

```bash
# Should fail (401 Unauthorized)
curl -X GET http://localhost:8000/api/v1/datasets

# Should succeed
curl -X GET http://localhost:8000/api/v1/datasets \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 5. Test Ownership

```bash
# User A creates resource
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -H "Authorization: Bearer USER_A_TOKEN" \
  -F "file=@data.csv"

# User B tries to access (should fail with 403 Forbidden)
curl -X GET http://localhost:8000/api/v1/datasets/{resource_id} \
  -H "Authorization: Bearer USER_B_TOKEN"
```

## Common Patterns

### Pattern 1: List Resources (User's Own)

```python
@router.get("/")
async def list_resources(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    # Filter by user_id to show only user's resources
    resources = db.query(Model).filter(Model.user_id == user_id).all()
    return resources
```

### Pattern 2: Get Single Resource with Ownership Check

```python
@router.get("/{resource_id}")
async def get_resource(
    resource_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    resource = db.query(Model).filter(Model.id == resource_id).first()

    if not resource:
        raise HTTPException(status_code=404, detail="Not found")

    verify_resource_ownership(resource.user_id, user_id, allow_admin=True, db=db)

    return resource
```

### Pattern 3: Create Resource

```python
@router.post("/")
async def create_resource(
    data: ResourceCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    # Automatically set user_id from auth
    resource = Model(
        **data.dict(),
        user_id=user_id
    )

    db.add(resource)
    db.commit()
    db.refresh(resource)

    return resource
```

### Pattern 4: Delete Resource

```python
@router.delete("/{resource_id}")
async def delete_resource(
    resource_id: str,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    resource = db.query(Model).filter(Model.id == resource_id).first()

    if not resource:
        raise HTTPException(status_code=404, detail="Not found")

    verify_resource_ownership(resource.user_id, user_id, allow_admin=True, db=db)

    db.delete(resource)
    db.commit()

    return {"message": "Deleted successfully"}
```

### Pattern 5: Update Resource

```python
@router.put("/{resource_id}")
async def update_resource(
    resource_id: str,
    data: ResourceUpdate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id)
):
    resource = db.query(Model).filter(Model.id == resource_id).first()

    if not resource:
        raise HTTPException(status_code=404, detail="Not found")

    verify_resource_ownership(resource.user_id, user_id, allow_admin=True, db=db)

    # Update fields
    for key, value in data.dict(exclude_unset=True).items():
        setattr(resource, key, value)

    db.commit()
    db.refresh(resource)

    return resource
```

## Migration Script

To help automate the updates, here's a Python script to check which files need updating:

```python
import re
from pathlib import Path

def find_mock_auth_files():
    """Find all endpoint files with mock authentication"""
    endpoint_dir = Path("backend/app/api/v1/endpoints")

    for file in endpoint_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue

        content = file.read_text()

        # Check for mock auth function
        if 'def get_current_user_id() -> str:' in content:
            print(f"\n{file.name}:")
            print("  ❌ Has mock auth function")

            # Count occurrences of user_id: str
            str_params = content.count('user_id: str')
            print(f"  📝 Found {str_params} 'user_id: str' parameters")

            # Check for endpoints without auth
            no_auth_pattern = r'async def \w+\([^)]*\):\s*(?!.*Depends\(get_current_user)'
            matches = re.findall(no_auth_pattern, content)
            if matches:
                print(f"  ⚠️  Possible endpoints without auth: {len(matches)}")

if __name__ == "__main__":
    find_mock_auth_files()
```

## Deployment Checklist

Before deploying to production:

- [ ] All mock auth functions removed
- [ ] All endpoints use real authentication
- [ ] Ownership verification added to resource endpoints
- [ ] Admin-only endpoints protected
- [ ] Database migration run successfully
- [ ] First admin user created
- [ ] SECRET_KEY changed from default
- [ ] Password requirements documented
- [ ] Token expiration configured
- [ ] HTTPS enabled
- [ ] CORS configured for production domains
- [ ] Rate limiting enabled on auth endpoints
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (not logging passwords/tokens)
- [ ] Frontend updated to handle auth flow
- [ ] Documentation updated
- [ ] API tests updated for authentication
- [ ] Integration tests run successfully

## Support

If you encounter issues while implementing authentication:

1. Check [AUTHENTICATION.md](AUTHENTICATION.md) for complete documentation
2. Review error messages in logs
3. Test with cURL to isolate issues
4. Verify database migration ran successfully
5. Confirm SECRET_KEY is set correctly
6. Check token expiration settings

For questions about specific endpoints or edge cases, refer to the implemented [datasets.py](backend/app/api/v1/endpoints/datasets.py) as a reference implementation.
