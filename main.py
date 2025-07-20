from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.8)

class GraphState(BaseModel):
    topic: str = Field(description="The topic for the graph.")
    detailed_report: Optional[str] = Field(default=None, description="A detailed report of the topic.")
    quiz_questions: Optional[str] = Field(default=None, desciorion="Quiz questions related to the topic.")
    summary: Optional[str] = Field(default=None, description="A summary of the topic.")

def content_generator(state: GraphState):
    topic = state.topic
    prompt = PromptTemplate(
        template="Generate a detailed report on the topic: {topic}.",
        input_variables=["topic"]
    )
    chain = prompt | llm
    response = chain.invoke({"topic": topic})
    return {"detailed_report": response.content}

def quiz_generator(state: GraphState):
    report = state.detailed_report
    prompt = PromptTemplate(
        template="Generate 5 quiz questions on the following report: {report}.",
        input_variables=["report"]
    )
    chain = prompt | llm
    response = chain.invoke({"report": report})
    return {"quiz_questions": response.content}


def summary_generator(state: GraphState):
    report = state.detailed_report
    prompt = PromptTemplate(
        template="Summarize the given report: {report}.",
        input_variables=["report"]
    )
    chain = prompt | llm
    response = chain.invoke({"report": report})
    return {"summary": response.content}

builder = StateGraph(GraphState)

builder.add_node("content_generator", content_generator)
builder.add_node("quiz_generator", quiz_generator)
builder.add_node("summary_generator", summary_generator)

builder.add_edge(START, "content_generator")
builder.add_edge("content_generator", "quiz_generator")
builder.add_edge("content_generator", "summary_generator")
builder.add_edge("quiz_generator", END)
builder.add_edge("summary_generator", END)

graph = builder.compile()

# response = graph.invoke({"topic": "F1 Racing"})

# print(response)

# print(graph.get_graph().draw_ascii())