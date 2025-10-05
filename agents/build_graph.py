from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents.state_schema import AgentState

def build_workflow(retriever_tool, agent, grade_documents, rewrite, generate):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {
        "tools": "retrieve",
        END: END,
    })
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()