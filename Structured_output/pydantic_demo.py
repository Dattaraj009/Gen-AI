from pydantic import BaseModel, EmailStr
from typing import Optional

class student(BaseModel):

    name: str
    age: Optional[int] = 32
    email: EmailStr

new_student = {'name':'datta','email':'abc@gmail.com'}

student = student(**new_student)

print(student)
