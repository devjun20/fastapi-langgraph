from typing import Optional, Dict, Any
from dotenv import load_dotenv
import asyncio
import json

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

# 검증 결과를 위한 Pydantic 모델
class ValidationResult(BaseModel):
    CoT: str = Field(description="Chain of thought reasoning process")
    is_valid: bool = Field(description="Whether the content meets the validation criteria")
    character_count: int = Field(description="Number of characters found in the content")
    feedback_message: str = Field(description="Specific feedback message for improvement")

class GraphState(BaseModel):
    topic: str = Field(description="The topic for the graph.")
    detailed_report: Optional[str] = Field(default=None, description="A detailed report of the topic.")
    quiz_questions: Optional[str] = Field(default=None, description="Quiz questions related to the topic.")
    summary: Optional[str] = Field(default=None, description="A summary of the topic.")
    retry_count: int = Field(default=0, description="Number of content generation retries.")
    validation_result: Optional[Dict[str, Any]] = Field(default=None, description="Validation result from the validator.")
    validation_feedback: Optional[str] = Field(default=None, description="Specific feedback from validation for content improvement.")

async def content_generator(state: GraphState):
    print("content_generator 실행")
    topic = state.topic
    
    # 재시도인 경우 validation 피드백을 활용한 프롬프트 사용
    if state.retry_count > 0 and state.validation_feedback:
        prompt = PromptTemplate(
            template="""Generate a detailed report on the topic (in english): {topic}.
            
            IMPORTANT FEEDBACK FROM VALIDATION:
            {validation_feedback}
            
            Please address the feedback above and improve your report accordingly.""",
            input_variables=["topic", "validation_feedback"]
        )
        response = await (prompt | llm).ainvoke({
            "topic": topic,
            "validation_feedback": state.validation_feedback
        })
    else:
        # 첫 번째 시도
        prompt = PromptTemplate(
            template="Generate a detailed report on the topic (in english): {topic}.",
            input_variables=["topic"]
        )
        response = await (prompt | llm).ainvoke({"topic": topic})
    
    return {"detailed_report": response.content}

async def validation_node(state: GraphState):
    print("validation_node 실행")
    report = state.detailed_report
    
    # JSON 출력 파서 설정
    parser = JsonOutputParser(pydantic_object=ValidationResult)
    
    prompt = PromptTemplate(
        template="""Analyze the following report and determine if it contains at least 50 different characters/people.

Report: {report}

Please provide your analysis in the following JSON format:
{format_instructions}

Count all types of characters including:
- Main characters
- Supporting characters  
- Historical figures
- Any named individuals mentioned
- Fictional characters
- Real people

Be thorough in your counting and reasoning.

IMPORTANT: In the feedback_message field, provide specific and actionable feedback:
- If validation passes: "Validation successful - found sufficient characters"
- If validation fails: Provide specific guidance like "Only found X characters, need Y more. Consider adding more supporting characters, background characters, or historical figures related to the topic. Specifically mention characters like [suggest specific types]"

Make the feedback_message as specific and helpful as possible for content improvement.""",
        input_variables=["report"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    response = await chain.ainvoke({"report": report})
    
    print(f"검증 결과: {response}")
    
    # validation_feedback를 GraphState에 저장
    feedback = response.get("feedback_message", "No specific feedback provided")
    
    return {
        "validation_result": response,
        "validation_feedback": feedback
    }

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

# 조건부 라우팅 함수
def decide_next_step(state: GraphState):
    """검증 결과에 따라 다음 단계를 결정"""
    validation_result = state.validation_result
    
    # 검증을 통과한 경우
    if validation_result and validation_result.get("is_valid", False):
        print("검증 통과! quiz_generator와 summary_generator로 진행")
        return "proceed_to_generators"
    
    # 검증에 실패했지만 재시도 횟수가 1회 미만인 경우
    elif state.retry_count < 1:
        print("검증 실패! content_generator로 재시도")
        print(f"피드백: {state.validation_feedback}")
        return "retry_content_generation"
    
    # 검증에 실패했고 재시도 횟수를 초과한 경우
    else:
        print("검증 실패하고 재시도 횟수 초과! 그래도 진행")
        return "proceed_to_generators"

def increment_retry_count(state: GraphState):
    """재시도 횟수 증가"""
    return {"retry_count": state.retry_count + 1}

async def fan_out(state: GraphState):
    return {}

# 그래프 구성
builder = StateGraph(GraphState)

# 노드 추가
builder.add_node("content_generator", content_generator)
builder.add_node("validation_node", validation_node)
builder.add_node("quiz_generator", quiz_generator)
builder.add_node("summary_generator", summary_generator)
builder.add_node("increment_retry", increment_retry_count)
builder.add_node("fan_out", fan_out)

# 엣지 설정
builder.add_edge(START, "content_generator")
builder.add_edge("content_generator", "validation_node")

# 조건부 엣지 - 검증 결과에 따른 라우팅
builder.add_conditional_edges(
    "validation_node",
    decide_next_step,
    {
        "retry_content_generation": "increment_retry",
        "proceed_to_generators": "fan_out"
    }
)
builder.add_edge("increment_retry", "content_generator")

builder.add_edge("fan_out", "quiz_generator")
builder.add_edge("fan_out", "summary_generator")


# 종료 엣지
builder.add_edge("quiz_generator", END)
builder.add_edge("summary_generator", END)

graph = builder.compile()

# 비동기 실행 함수
async def main():
    # 동시 요청 처리
    task1 = graph.ainvoke({"topic": "케로로 소대"})
    
    
    # 요청을 실행하고 결과를 기다림
    response = await asyncio.gather(task1)
    
    print("=== 최종 결과 ===")
    print(response)


    final_state = GraphState(**response[0])

    print(f"재시도 횟수: {final_state.retry_count}")          # ✅
    print(f"검증 피드백: {final_state.validation_feedback}")  # ✅
    print(f"검증 결과: {final_state.validation_result}")      # ✅




def draw():
    print(graph.get_graph().draw_ascii())
    

# 비동기 실행
if __name__ == "__main__":
    draw()
    asyncio.run(main())
    # draw()