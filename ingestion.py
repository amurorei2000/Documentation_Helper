from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader, FireCrawlLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# HTML 문서 파싱
def ingest_docs():
    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(raw_documents)
    for doc in docs:
        new_url = doc.metadata["source"].replace("\\", "/")
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(docs)} to Pinecone")

    PineconeVectorStore.from_documents(
        documents=docs, embedding=embeddings, index_name="langchain-doc-index"
    )


# Fire Crawl을 이용한 웹 페이지 -> Mark down 크롤링
def ingest_docs_with_firecrawl() -> None:
    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/tutorials/",
        "https://python.langchain.com/docs/how_to/",
        "https://python.langchain.com/docs/concepts/",
    ]

    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "crawlerOptions": {"limit": 5},
                "pageOptions": {"onlyMainContent": True},
                "wait_until_done": True,
            },
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} to Pinecone")

        PineconeVectorStore.from_documents(
            documents=docs, embedding=embeddings, index_name="firecrawl-index"
        )
        print(f"***Loading {url} to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
    # ingest_docs_with_firecrawl()
