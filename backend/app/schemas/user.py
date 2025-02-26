from typing import Optional

from pydantic import BaseModel, EmailStr, UUID4


class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    full_name: Optional[str] = None


class UserCreate(UserBase):
    email: EmailStr
    password: str


class UserUpdate(UserBase):
    password: Optional[str] = None


class UserInDBBase(UserBase):
    id: Optional[UUID4] = None

    class Config:
        from_attributes = True


class User(UserInDBBase):
    pass
