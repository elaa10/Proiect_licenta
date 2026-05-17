from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

# Ce primim cand utilizatorul se inregistreaza
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)

# Ce primim cand utilizatorul se autentifica
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Ce returnam catre frontend (FARA parola)
class UserResponse(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True  # permite conversie din obiect SQLAlchemy

# Raspunsul la /login: tokenul JWT
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"