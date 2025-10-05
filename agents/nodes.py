from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode

def agent_node(tools):
    def agent(state):
        messages = state["messages"]
        question = messages[0].content
        model = OllamaLLM(model="llama3.2", temperature=0, streaming=True)

        if "search" in question.lower():
            tool_executor = ToolNode(tools)
            result = tool_executor.invoke({"input": question})
            return {"messages": [result]}
        else:
            response = model.invoke(messages)
            return {"messages": [response]}
    return agent

def grade_documents_node():
    class GradeSchema(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

    def grade_documents(state):
        model = OllamaLLM(model="llama3.2", temperature=0, streaming=True)
        llm_with_tool = model.with_structured_output(GradeSchema)

        prompt = PromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.
            Context:
            {context}

            Question:
            {question}

            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            Give a binary score 'yes' or 'no'.
"""
        )

        question = state["messages"][0].content
        docs = state["messages"][-1].content
        result = (prompt | llm_with_tool).invoke({"question": question, "context": docs})
        return "generate" if result.binary_score == "yes" else "rewrite"
    return grade_documents

def rewrite_node():
    def rewrite(state):
        question = state["messages"][0].content
        msg = [HumanMessage(content=f"""
        Look at the input and try to reason about the underlying semantic intent.
        Here is the initial question:
        -------
        {question}
        -------
        Formulate an improved question:
        """)]
        model = OllamaLLM(model="llama3.2", temperature=0, streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}
    return rewrite

def generate_node():
    def generate(state):
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate.from_template("""
        You are an expert assistant. Use only the provided context to answer the question.
        If the context does not contain the answer, say "I don't know based on the documents."

        Context:
        {context}

        Question:
        {question}
        """)

        llm = OllamaLLM(model="llama3.2", temperature=0, streaming=True)
        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    return generate