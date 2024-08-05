# InternLM

![InternLM工作流程](../image_resources/internlm_process.png)

- 增量续训：
  - 使用场景：让基座模型学到一些新知识
  - 训练数据：细分领域的文章、书籍、代码
- 有监督微调：
  - 使用场景：让模型学会理解各种指令进行对话；
  - 训练数据：高质量的对话、问答数据


## 提示工程

- 提示工程是一种通过设计和调整输入（Prompts）来改善模型性能或控制其输出结果的技术；
- 模型回复过程中，首先获取用户输入的文本，然后处理文本特征，并根据输入的文本特征预测后续的文本（Next Token Prediction）
- 有六大基本原则：
  1. 指令要清晰
  2. 提供参考内容
  3. 将复杂的任务拆分成子任务
  4. 给 LLM 思考过程
  5. 使用外部工具
  6. 系统性测试变化  ？？

- **Prompt**
  - 广义提示词，一切影响模型输出结果的内容（即会作为模型输入的内容）都被视为Prompt

- **System Prompt**
  - 指每次对话开头自带的一段文本，也叫**开头提示词**

### LangGPT 结构化提示词

- LangGPT  Language For GPT-like LLMs  结构化提示词
- 是一个帮助编写高质量提示词的工具
- 一个完整的提示词包含：模块 + 内部元素
  - 模块：要求， 提示
  - 内部元素：赋值型，方法型


## RAG

- RAG
  - `Retrieval Augmented Generation` 检索增强生成
  - ![LLM_RAG](../image_resources/LLM_RAG.png)
  1. 将自己专业领域的文档编码成向量，组成向量库；
  2. 将用户的询问编码成向量；
  3. 使用用户询问的向量在向量库中检索相关的文档块；
  4. 将检索到的文档块与原始问题一起作为提示，输入到LLM中，生成最终的回答；

- 案例: 解析论文回答问题
- 选用论文：CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models
- 不使用RAG的示例代码：
  ```python
  from llama_index.llms.huggingface import HuggingFaceLLM
  from llama_index.core.llms import ChatMessage
  
  llm = HuggingFaceLLM(
    model_name = "/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",
    tokenizer_name="/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
  )

  rsp = llm.chat(messages=[ChatMessage(content="介绍模型'CatVTON'可以运用在什么场景，有什么用处，能达到什么效果")])
  print(rsp)
  ```
- 使用RAG的示例代码：
  
  ```python
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.llms.huggingface import HuggingFaceLLM
  from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

  # 用于将文本转换为向量表示
  embed_model = HuggingFaceEmbedding(
    model_name = "/root/tasks/task_L1/03LlamaIndex/model/sentence-transformer/"
    )

  # 将创建的Embed模型赋值给全局的embed_model属性
  Settings.embed_model = embed_model

  llm = HuggingFaceLLM(
    model_name="/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",
    tokenizer_name="/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
    )
  Settings.llm = llm

  # 从指定目录读取所有文档，并加载数据到内存中
  documents = SimpleDirectoryReader("/root/tasks/task_L1/03LlamaIndex/data/").load_data()
  # 创建一个VectorStoreIndex，并使用之前加载的文档来构建索引，此索引将文档转换为向量，并存储这些向量以便快速检索
  index = VectorStoreIndex.from_documents(documents)
  # 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应
  query_engine = index.as_query_engine()
  response = query_engine.query("介绍模型'CatVTON'可以运用在什么场景，有什么用处，能达到什么效果")
  print(response)
   ```
 
- 使用RAG前
- ![non-use RAG](../image_resources/non-use-rag.png)
- 使用RAG后
- ![use RAG](../image_resources/use-rag.png)

## Finetune

- 在大模型的下游应用中，常用到的两种微调模式：**增量预训练** 和 **指令跟随**
  - 增量预训练：在已有预训练模型（例如：InternLM基座模型）的基础上，利用特定领域的数据进行进一步训练的方法。目的是在保持模型原有能力的同时，注入新的领域知识，进一步优化现有的预训练模型，从而提升模型在特定领域任务中的表现；
  - 指令跟随：是指让模型根据用户输入的指令来执行相应的操作。模型通过对大量自然语言指令和相应的操作的数据进行训练，学习如何将指令分解为具体的子任务，并选择合适的模块来执行这些任务；
- 常用的微调技术有：LoRA 和 QLoRA

### LoRA

- LoRA (Low-Rank Adaptation)
- 使用低精度权重对大型预训练语言模型进行微调的技术；
- 核心思想是，在不改变原有模型权重的情况下，通过添加少量新参数来微调
- 这种方法降低了模型的存储需求，也降低了计算成本，实现了对大模型的快速适应；

### QLoRA

- QLoRA (Quantized LoRA)
- 是对LoRA的一种改进，它通过引入高精度权重和可学习的低秩适配器来提高模型的准确性。
- 在LoRA的基础上，引入了量化技术。通过将预训练模型量化为int4格式，可以进一步减少微调过程中的计算量，同时也可以减少模型的存储空间，这对于在资源有限的设备上运行模型非常有用。最终，可以使我们在消费级的显卡上进行模型的微调训练

## LMDeploy

- LMDeploy 是一个用于压缩、部署 LLM 的工具包，核心功能：
  - 高效推理
  - 有效量化：支持权重量化和 k/v 量化，4bit推理性能是FP16的2.4倍

## 问题：
1. 如果要构建一个对话智能体，需要构建多大的数据集？
2. 怎么为下游任务微调？
3. 能不能给一些专业领域的文档，后续就可以直接提问了？
4. 