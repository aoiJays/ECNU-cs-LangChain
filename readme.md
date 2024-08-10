# 基于LangChain和大语言模型的本地知识库问答

[TOC]

## 数据采集

- [华东师范大学计算机科学与技术学院 - 学院概况 - 学院简介](http://www.cs.ecnu.edu.cn/xyjj/list.htm)
    - `data/学院简介.md`

- [华东师范大学计算机科学与技术学院 - 学院概况 - 历史沿革](http://www.cs.ecnu.edu.cn/lsyg/list.htm)
    - `data/历史沿革.md`
- [华东师范大学计算机科学与技术学院 - 人才培养 - 培养方案](http://www.cs.ecnu.edu.cn/pyfa/list.htm)
    - `data/2023级计算机科学与技术学术硕士研究生培养方案.pdf`

## 数据处理

文档存储地址：

```python
# 定义文件路径
pdf_file = "./data/2023级计算机科学与技术学术硕士研究生培养方案.pdf"
md_file1 = "./data/历史沿革.md"
md_file2 = "./data/学院简介.md"

```

langchain内置了对pdf、markdown进行读取的封装

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 加载文件内容
pdf_loader = PyPDFLoader(file_path=pdf_file)
md_loader1 = UnstructuredMarkdownLoader(file_path=md_file1)
md_loader2 = UnstructuredMarkdownLoader(file_path=md_file2)
```



此时我们需要对文档进行切分，得到若干个chunk

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
def split(chunk_size, doc):
    text_splitter = RecursiveCharacterTextSplitter(
    	chunk_size = chunk_size,
    	chunk_overlap  = chunk_size // 10,
	)
    return text_splitter.split_documents(doc)
# 设定overlap率在10%

# 切分文档 - 根据实际段落情况进行切分
pdf_documents = split(chunk_size=250, doc=pdf_loader.load()) # 培养方案段落更长

# 两个md文档的单句长度不太一致 具体分析
md_documents1 = split(chunk_size=50, doc=md_loader1.load())
md_documents2 = split(chunk_size=100, doc=md_loader2.load())

# 合并
documents = pdf_documents + md_documents1 + md_documents2
```



在实际操作中，发现有一些问题本身提供的信息非常少，很难被检索出来

但是结合实际文档，我们发现日期是一个非常敏感的信息

```python
...
2003年：增设系统科学一级学科博士点和计算机系统结构专业硕士点；
2006年：增设计算机科学与技术一级学科硕士点和计算机应用技术二级学科博士点；
2010年：增设计算机科学与技术一级学科博士点；
2012年：增设计算机科学与技术一级学科博士后流动站；
...
```



因此对`xxxx年x月x日`或`xxxx年`格式的日期，使用正则表达式进行检索

将日期字符串重复拼接在原字符串最后

实现加大权重的效果



```python
import re
# 正则表达式匹配特定日期格式
date_patterns = [
    r'\d{4}年\d{1,2}月\d{1,2}日',
    r'\d{4}年'
]

# 通过重复日期来增加权重
def weight_dates(text, weight_factor=3):
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        ok = 0
        for date in dates:
            text += ' ' + date * weight_factor 
            ok = 1
        if ok : break
    return text

# 日期加权
for i in range(len(documents)):
    documents[i].page_content = weight_dates(documents[i].page_content)

```



## 向量数据库

使用FAISS

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings

# 调用词嵌入向量API
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
)

# 构建数据库 使用FAISS
db = FAISS.from_documents(documents, embeddings) 
retriever = db.as_retriever( # 转换为检索器 返回最相关的k个文档
    search_kwargs = {
        'k': 10
	}
)
```



## 大模型

```python
from langchain_community.llms import Tongyi
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Tongyi( 
	# model_name="qwen1.5-1.8b-chat",  # 测试用 免费
	model_name="qwen2-72b-instruct", # 效果最好 但是氪金
	temperature=0.95, top_p=0.7,
    streaming=True, callbacks=[StreamingStdOutCallbackHandler()] # 实现流式输出
)
    
```



## 检索链

```python
from langchain.chains import RetrievalQA

    # 构建检索链
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 修改prompt 防止输出过多内容 (省钱)
    # 以及大部分问题都是一个陈述句即可完成回答 言多必失
    # 添加内容 'by a short sentence no more than 20 words.'
    qa.combine_documents_chain.llm_chain.prompt.template = '''
Use the following pieces of context to answer the question directly at the end by a short sentence no more than 20 words.if you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Answer:
'''
```



调用方式：

```python
qa.invoke(weight_dates(query))
```

- 对query进行同样的日期信息加权
- `qa`调用`retriever`，对FAISS数据库进行检索，得到相似度最高的k个文档，填充在`{context}`中
- `query`填充在`{question}`中
- 将模板生成的prompt喂给大模型进行回答



## 评价

### 测试数据准备

直接文件读入即可，全部放在`examples`列表中，以字典形式

```python
    # 读取题目
    question_path = './materials/测试题目.md'
    answers_path = './materials/测试答案.md'

    questions = []
    with open(question_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 1: questions.append(line)

    answers = []
    with open(answers_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0: answers.append(line)


    # 对齐问题 答案 准备测试数据
    num_questions = len(questions)
    examples = [{
        'query': questions[i],
        'answer': answers[i]
    } for i in range(num_questions)]
```



格式：

```python
[
    {
        'query': '华东师范大学计算机科学与技术学院的前身是什么？', 
        'answer': '华东师范大学计算机科学与技术学院的前身是1979年创建的计算机科学系'
    }, 
    {
        'query': '计算机科学与技术学院现有正高职称多少人？', 
        'answer': '计算机科学与技术学院现有正高职称27人'
    },
]
```



在进行输入前，同样需要将所有问题进行加权处理：

```python
    # 处理输入
    for i in range(num_questions):
        examples[i]['query'] = weight_dates(examples[i]['query'])
```



### 测试链

我很难通过字符串比较的方式评价大模型的回答是否正确

因此使用大模型进行打分

构建`QAEvalChain`，将问题、正确答案、预测答案喂给大模型

由大模型判断`CORRECT`或`INCORRECT`



实际情况中，我们发现大模型在打分时过于严厉

- 官网采集数据：`1999年：“计算机科学系”更名为”计算机科学技术系”；`
- 测试数据：`{'query': '1999年，计算机科学系更名为什么？', 'answer': '计算机科学与技术系'},`

此时大模型回答根据文档信息中的`计算机科学技术系`进行回答

但是有时候大模型评估时，认为`计算机科学技术系`和`计算机科学与技术系`不相符，打出了`INCORRECT`



因此我在模板中添加如下：

```
 Allow for some incomplete expressions of related nouns.
```

代码：

```python
from langchain.evaluation.qa import QAEvalChain
# 测试链
eval_chain = QAEvalChain.from_llm(llm)

# 模糊名词
eval_chain.prompt.template = '''
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Allow for some incomplete expressions of related nouns. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:
'''
```



开始评估：

```python
graded_outputs = eval_chain.evaluate(examples, predictions)

# 打印结果 计算正确率
correct = 0
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]["query"])
    print("Real Answer: " + predictions[i]["answer"])
    print("Predicted Answer: " + predictions[i]["result"])
    print("Predicted Grade: " + graded_outputs[i]["results"])
    print()
    
    # 有时候除了CORRECT，会偶尔在后面继续说话
    # 所以判断前7个字符
    if graded_outputs[i]["results"][:7].strip() == 'CORRECT': correct += 1

    
    
print(f'Accuracy = {correct/num_questions * 100}%')
```



## 结果

> acc: 100%
见`res.txt`




