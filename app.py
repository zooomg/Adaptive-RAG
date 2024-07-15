from typing import List

from fastapi import FastAPI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

from langserve import add_routes

from Graph.Graph import graph

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

class Input(BaseModel):
    question: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

# Pydantic Schema 정의 (API 출력값)
class Output(BaseModel):
    output: str

# 라우팅 실제로 추가하기
add_routes(
    app,
    graph.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

# .py 파일 실행 (uvicorn)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)