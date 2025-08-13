from typing import Optional
from dotenv import load_dotenv
import asyncio

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemma-3-12b-it",
    google_api_key=google_api_key,
    temperature=0.7
)

final_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=anthropic_api_key,
    temperature=0.7
)

class GraphState(BaseModel):
    topic: str = Field(description="The topic for the graph.")
    detailed_report: Optional[str] = Field(default=None, description="A detailed report of the topic.")
    quiz_questions: Optional[str] = Field(default=None, description="Quiz questions related to the topic.")
    summary: Optional[str] = Field(default=None, description="A summary of the topic.")

async def content_generator(state: GraphState):
    print("content_generator 실행")
    topic = state.topic
    prompt = PromptTemplate(
        template="Generate a detailed report on the topic (in english): {topic}.",
        input_variables=["topic"]
    )
    chain = prompt | llm
    response = await chain.ainvoke({"topic": topic})
    return {"detailed_report": response.content}

async def quiz_generator(state: GraphState):
    print("quiz_generator 실행")
    report = state.detailed_report
    prompt = PromptTemplate(
        template="Generate 5 quiz questions on the following report (한국어로): {report}.",
        input_variables=["report"]
    )
    chain = prompt | llm
    response = await chain.ainvoke({"report": report})
    return {"quiz_questions": response.content}

async def summary_generator(state: GraphState):
    print("summary_generator 실행")
    report = state.detailed_report
    prompt = PromptTemplate(
        template="Summarize the given report (한국어로): {report}.",
        input_variables=["report"]
    )
    chain = prompt | final_llm
    response = await chain.ainvoke({"report": report})
    return {"summary": response.content}

# 그래프 구성
builder = StateGraph(GraphState)

builder.add_node("content_generator", content_generator)
builder.add_node("quiz_generator", quiz_generator)
builder.add_node("summary_generator", summary_generator)

# 병렬로 실행되도록 수정
builder.add_edge(START, "content_generator")
builder.add_edge("content_generator", "quiz_generator")
builder.add_edge("content_generator", "summary_generator")
builder.add_edge("quiz_generator", END)
builder.add_edge("summary_generator", END)

graph = builder.compile()

# 비동기 실행 함수
async def main():
    # 동시 요청 처리
    task1 = graph.ainvoke({"topic": "케로로 소대"})
    task2 = graph.ainvoke({"topic": "진격의 거인"})
    
    # 두 요청을 병렬로 실행하고 결과를 기다림
    response1, response2 = await asyncio.gather(task1, task2)
    
    print("=== 케로로 소대 결과 ===")
    print(response1)
    print("\n=== 진격의 거인 결과 ===")
    print(response2)

# 비동기 실행
if __name__ == "__main__":
    asyncio.run(main())