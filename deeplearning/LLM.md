# Ollama

## 环境搭建

- ollama 服务下载
  - [官网](https://ollama.com/)
  - 下载安装
  - 测试：`ollama help`
  - `ollama run llama3.1:8b` 下载并运行相应的大模型服务
  - `ollama list`  查看本地已有的大模型
  - `/bye`  退出交互环境

- requirements
  - `pip install ollama langchain_core langchain_ollama langchain_community langchain_chroma`

```
aiohappyeyeballs==2.4.4
aiohttp==3.11.9
aiosignal==1.3.1
alembic==1.14.0
annotated-types==0.7.0
anthropic==0.39.0
anyio==4.6.2.post1
arxiv==2.1.3
asgiref==3.8.1
async-timeout==4.0.3
attrs==24.2.0
autogenstudio==0.1.5
backoff==2.2.1
bcrypt==4.2.1
build==1.2.2.post1
cachetools==5.5.0
certifi==2024.8.30
chardet==3.0.4
charset-normalizer==3.4.0
chroma-hnswlib==0.7.6
chromadb==0.5.23
click==8.1.7
colorama==0.4.6
coloredlogs==15.0.1
dataclasses-json==0.6.7
Deprecated==1.2.15
diskcache==5.6.3
distro==1.9.0
django-environ==0.4.5
docker==7.1.0
docstring_parser==0.16
durationpy==0.9
eval_type_backport==0.2.0
exceptiongroup==1.2.2
fastapi==0.115.5
feedparser==6.0.11
filelock==3.16.1
filetype==1.0.5
FLAML==2.3.2
flatbuffers==24.3.25
frozenlist==1.5.0
fsspec==2024.10.0
google-ai-generativelanguage==0.6.10
google-api-core==2.23.0
google-api-python-client==2.153.0
google-auth==2.36.0
google-auth-httplib2==0.2.0
google-cloud-aiplatform==1.72.0
google-cloud-bigquery==3.27.0
google-cloud-core==2.4.1
google-cloud-resource-manager==1.13.1
google-cloud-storage==2.18.2
google-crc32c==1.6.0
google-generativeai==0.8.3
google-resumable-media==2.7.2
googleapis-common-protos==1.66.0
greenlet==3.1.1
grpc-google-iam-v1==0.13.1
grpcio==1.68.0
grpcio-status==1.68.0
h11==0.14.0
httpcore==1.0.7
httplib2==0.22.0
httptools==0.6.4
httpx==0.27.2
httpx-sse==0.4.0
huggingface-hub==0.26.3
humanfriendly==10.0
idna==3.10
ijson==2.6.1
importlib_metadata==8.5.0
importlib_resources==6.4.5
iso8601==0.1.12
jiter==0.7.1
jsonpatch==1.33
jsonpath-python==1.0.6
jsonpointer==3.0.0
kubernetes==31.0.0
langchain==0.3.9
langchain-chroma==0.1.4
langchain-community==0.3.9
langchain-core==0.3.21
langchain-ollama==0.2.1
langchain-text-splitters==0.3.2
langsmith==0.1.147
loguru==0.7.2
Mako==1.3.6
markdown-it-py==3.0.0
marshmallow==3.23.1
mdurl==0.1.2
mistralai==1.2.2
mmh3==5.0.1
monotonic==1.6
mpmath==1.3.0
multidict==6.1.0
mypy-extensions==1.0.0
numpy==1.26.4
oauthlib==3.2.2
ollama==0.4.2
onnxruntime==1.20.1
openai==1.54.4
opentelemetry-api==1.28.2
opentelemetry-exporter-otlp-proto-common==1.28.2
opentelemetry-exporter-otlp-proto-grpc==1.28.2
opentelemetry-instrumentation==0.49b2
opentelemetry-instrumentation-asgi==0.49b2
opentelemetry-instrumentation-fastapi==0.49b2
opentelemetry-proto==1.28.2
opentelemetry-sdk==1.28.2
opentelemetry-semantic-conventions==0.49b2
opentelemetry-util-http==0.49b2
orjson==3.10.12
overrides==7.7.0
packaging==24.2
posthog==3.7.4
propcache==0.2.1
proto-plus==1.25.0
protobuf==5.28.3
psycopg==3.2.3
psycopg2==2.9.6
pyasn1==0.6.1
pyasn1_modules==0.4.1
pyautogen==0.3.2
pydantic==2.9.2
pydantic-settings==2.6.1
pydantic_core==2.23.4
Pygments==2.18.0
pyparsing==3.2.0
PyPika==0.48.9
pyproject_hooks==1.2.0
pyreadline3==3.5.4
python-dateutil==2.8.2
python-dotenv==1.0.1
python-http-client==3.3.7
python3-openid==3.2.0
pywin32==308
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
rich==13.9.4
rsa==4.9
sgmllib3k==1.0.0
shapely==2.0.6
shellingham==1.5.4
six==1.16.0
sniffio==1.3.1
SQLAlchemy==2.0.36
sqlmodel==0.0.22
sqlparse==0.4.4
starlette==0.41.2
sympy==1.13.3
tenacity==9.0.0
termcolor==2.5.0
tiktoken==0.8.0
tokenizers==0.20.3
tomli==2.2.1
tqdm==4.67.0
typer==0.13.0
typing-inspect==0.9.0
typing_extensions==4.12.2
tzdata==2024.2
unicodecsv==0.14.1
uritemplate==4.1.1
urllib3==2.2.3
uvicorn==0.32.0
waitress==1.4.3
watchfiles==1.0.0
websocket-client==1.8.0
websockets==14.1
whitenoise==5.0.1
win32-setctime==1.1.0
wrapt==1.17.0
yarl==1.18.3
zipp==3.21.0

```


## 对话

```python
import ollama
from ollama import Client
from openai import OpenAI

if __name__ == "__main__":
    
    llm = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="sk-1234567890",
    )
    response = llm.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": "Hello, what is your name!"},
        ])
    
    print(response.choices[0].message.content)
    
    exit(0)
    
    client = Client(host="http://localhost:11434")
    response = client.chat(model="llama3.2:3b",
                           messages=[
                               {"role": "user", "content": "Hello, world!"},
                           ])
    print(response["message"]["content"])
    
    exit(0)

    result = ollama.generate(model="llama3.2:3b",
                    prompt="Hello, world!",)
    print(result["response"])
    
    response = ollama.chat(model="llama3.2:3b",
                           messages=[
                               {"role": "user", "content": "Hello, world!"},
                           ])
    print(response["message"]["content"])
    
```

## Langchain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that gives a one-line definition of the word entered by user"),
        ("human", "{user_input}"),
    ]
)

message = chat_template.format_messages(user_input="happy")
print(message)

llm = ChatOllama(model="llama3.2:3b", temperature=0.9)
ai_msg = llm.invoke(message)
print(ai_msg)

chain = chat_template | llm | StrOutputParser()
print(chain.invoke({"user_input": "happy"}))
```

## RAG

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

file = "./data/paddlex_pipeline_guide.md"
raw_documents = TextLoader(file, encoding="utf-8").load()
# print(raw_documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=20)
documents = text_splitter.split_documents(raw_documents)
print(f"document len:{len(documents)}") 
print(documents[0])

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
db = Chroma.from_documents(
    documents, oembed, persist_directory="./chroma_db"
    )

query = "what is PaddleX Pipeline?"
docs = db.similarity_search(query)
print(f"get {len(docs)} docs")
# print(docs[0].page_content)


template = """
Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2:3b", temperature=0.9)
retriver = db.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
print("==="*5)
print(chain.invoke("how can I use paddlex?"))


```

