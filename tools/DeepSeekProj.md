# DeepSeek使用

## 基础

### 命令行使用
- 安装ollama  `>>ollama --version`
- `>>ollama pull deepseek-r1`
- `>>ollama run deepseek-r1`  开启对话进程

### Python使用
- 可以使用`requests`库，访问本地api `http://localhost:11434/api/generate`

## 自然语言文本处理
- 文本总结
- 文本生成
- 纠正拼写错误和语法错误  `prompt=f"Correct my grammar and spelling mistakes in the following text:\n\n {text}"`
- 命名实体抽取 `prompt = f"Extract all names entities(persons, organizations, locations, dates) from:\n\n {text}"`
- 文本情感分析  `prompt = f"Classify the sentiment of the following text as Positive, Negative, or Neutral:\n\n {text}"`

- 实例代码：
```python
import requests
import gradio as gr
from fastapi import FastAPI

'''
- 使用gradio可以创建web应用
- 使用fastapi可以创建API  使用 uvicorn 启动服务 `uvicorn proj01:app --reload`
'''

OLLAMA_URL = r"http://localhost:11434/api/generate"

# app = FastAPI()  # 用于创建API  uvicorn proj01:app --reload
# @app.post("/summarize/")
def summarize_text(text: str):
    prompt = f"Summarize the following text in Chinese: \n\n{text}"

    payload = {
        "model":"deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No Summary Generated")
    else:
        return f"Error: {response.text}"

interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(label="Original Text", lines=10, placeholder="Enter text to summarize"),
    outputs=gr.Textbox(label="Summarized Text"),
    title="AI-Powered Text Summarizer",
    description="Enter a long text and DeepSeek AI will generate a concise summary"
)

if __name__ == "__main__":
    interface.launch()
    exit()

    sample_text = '''
    Image histograms capture the way a scene is rendered using the available pixel intensity values. By analyzing the distribution of the pixel values over an image, it is possible to use this information to modify and possibly improve an image. This recipe explains how you can use a simple mapping function, represented by a lookup table, to modify the pixel values of an image. As we will see, lookup tables are often defined from histogram distributions.
    '''
    print(summarize_text(sample_text))
    print("done")
```

## 聊天机器人


## 自动化


## 辅助编程


## 商业数据分析

