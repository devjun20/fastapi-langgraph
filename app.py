from main import graph
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

class APIInput(BaseModel):
    topic: str = Field(description="Input topic for the api endpoint.")

@app.post("/chat")
def chat(input: APIInput):
    response = graph.invoke({"topic": input.topic})
    return response