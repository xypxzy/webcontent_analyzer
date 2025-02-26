from typing import Optional

from pydantic import BaseModel, UUID4


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[UUID4] = None
    exp: Optional[int] = None
