"""
Authentication endpoints: login, register, refresh token, password management
"""
from datetime import datetime, timedelta
from typing import Any
import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
    get_current_user,
    get_current_user_id,
    decode_token,
)
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserLogin,
    UserRead,
    Token,
    RefreshToken,
    PasswordChange,
    APIKeyResponse,
)

router = APIRouter()


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Register a new user account.

    Args:
        user_data: User registration data (email, password)
        db: Database session

    Returns:
        The created user object

    Raises:
        HTTPException: If email already exists
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        id=uuid.uuid4(),
        email=user_data.email,
        password_hash=hashed_password,
        is_active=True,
        is_admin=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post("/login", response_model=Token)
def login(
    login_data: UserLogin,
    db: Session = Depends(get_db)
) -> Any:
    """
    User login endpoint. Returns JWT access and refresh tokens.

    Args:
        login_data: User login credentials (email, password)
        db: Database session

    Returns:
        JWT access and refresh tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    # Authenticate user
    user = authenticate_user(db, login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Token)
def refresh_access_token(
    refresh_data: RefreshToken,
    db: Session = Depends(get_db)
) -> Any:
    """
    Refresh access token using refresh token.

    Args:
        refresh_data: Refresh token
        db: Database session

    Returns:
        New JWT access and refresh tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    # Decode refresh token
    try:
        payload = decode_token(refresh_data.refresh_token)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify it's a refresh token
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user ID
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user exists and is active
    user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new tokens
    access_token = create_access_token(data={"sub": str(user.id)})
    new_refresh_token = create_refresh_token(data={"sub": str(user.id)})

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserRead)
def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user's profile information.

    Args:
        current_user: The authenticated user from JWT token

    Returns:
        Current user's profile data
    """
    return current_user


@router.post("/change-password", status_code=status.HTTP_200_OK)
def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Change user's password.

    Args:
        password_data: Current and new passwords
        current_user: The authenticated user
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If current password is incorrect
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )

    # Update password
    current_user.password_hash = get_password_hash(password_data.new_password)
    current_user.updated_at = datetime.utcnow()
    db.commit()

    return {"message": "Password changed successfully"}


@router.post("/api-key/generate", response_model=APIKeyResponse)
def generate_api_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Generate a new API key for programmatic access.
    This will invalidate any existing API key.

    Args:
        current_user: The authenticated user
        db: Database session

    Returns:
        The generated API key
    """
    # Generate secure random API key
    api_key = f"aip_{secrets.token_urlsafe(32)}"

    # Update user's API key
    current_user.api_key = api_key
    current_user.updated_at = datetime.utcnow()
    db.commit()

    return {
        "api_key": api_key,
        "user_id": current_user.id,
        "created_at": datetime.utcnow()
    }


@router.delete("/api-key/revoke", status_code=status.HTTP_200_OK)
def revoke_api_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Revoke (delete) the user's API key.

    Args:
        current_user: The authenticated user
        db: Database session

    Returns:
        Success message
    """
    current_user.api_key = None
    current_user.updated_at = datetime.utcnow()
    db.commit()

    return {"message": "API key revoked successfully"}


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Logout endpoint (for client-side token cleanup).
    Since JWTs are stateless, actual logout happens client-side.

    Args:
        current_user: The authenticated user

    Returns:
        Success message
    """
    return {"message": "Logged out successfully. Please delete your tokens."}
