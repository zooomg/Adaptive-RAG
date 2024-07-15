### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)