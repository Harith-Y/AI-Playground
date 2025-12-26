from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class UserBase(BaseModel):
    email: str  # TODO: Change to EmailStr when email-validator is installed


class UserCreate(UserBase):
    pass


class UserRead(UserBase):
    id: UUID
    created_at: datetime | None = None

    class Config:
        from_attributes = True
