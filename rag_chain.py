from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableMap
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import os

def get_expression_chain(
    retriever
) -> Runnable:
    """Return a chain defined primarily in LangChain Expression Language"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n3. If the answer is not present in the context, say 'I could not find the answer."),
            ("human", "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "),
        ]
    )

    # Initialize the ChatOpenAI object with GPT-3.5 Turbo model and set the temperature parameter
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = AzureChatOpenAI(
        azure_endpoint="https://rajrishabhrutuja.openai.azure.com/",
        azure_deployment="rishabh-gpt35-turbo",
        api_version="2024-02-01",
        api_key=os.environ.get("OPENAI_KEY"),
        temperature=0
    )

    
    def format_docs(docs):
        return "\n\n".join(doc[0].page_content for doc in docs)

    def similarity_func(params):
        return retriever.similarity_search_with_score(**params)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context_str=(lambda x: format_docs(x["context_str"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context_str": similarity_func, "query_str": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_with_source