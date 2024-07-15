from langgraph.graph import END, StateGraph, START

from .GraphState import GraphState
from .GraphFlow import (
    i_dont_know, 
    retrieve, 
    grade_documents, 
    generate, 
    transform_query, 
    route_question, 
    decide_to_generate,
    grade_generation_v_documents_and_question
)

workflow = StateGraph(GraphState)

# Define the nodes
# workflow.add_node("web_search", web_search)  # web search
workflow.add_node("i_dont_know", i_dont_know)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "i_dont_know",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("i_dont_know", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
graph = workflow.compile()