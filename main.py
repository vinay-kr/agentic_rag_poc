from agents.load_documents import load_and_split_documents
from agents.vectorstore_setup import setup_vectorstore
from agents.nodes import agent_node, grade_documents_node, rewrite_node, generate_node
from agents.build_graph import build_workflow
from langchain_core.messages import HumanMessage
import pprint

# Step 1: Load and split documents
urls = [
    "https://hbr.org/2021/01/what-you-were-taught-about-happiness-isnt-true",
    "https://buffer.com/resources/be-happy-today/",
]
doc_splits = load_and_split_documents(urls)

# Step 2: Setup vectorstore and retriever tool
retriever_tool = setup_vectorstore(doc_splits)

# Step 3: Build nodes
agent = agent_node([retriever_tool])
grade_documents = grade_documents_node()
rewrite = rewrite_node()
generate = generate_node()

# Step 4: Build and compile graph
graph = build_workflow(retriever_tool, agent, grade_documents, rewrite, generate)

# Step 5: Run the graph
inputs = {
    "messages": [
        HumanMessage(content="What are the 5 pointers to make a human live happy life?")
    ]
}

for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")