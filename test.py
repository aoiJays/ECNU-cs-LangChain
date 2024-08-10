from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
import os
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
import re
import time

# 文档切分
def split(chunk_size, doc):
    text_splitter = RecursiveCharacterTextSplitter(
    	chunk_size = chunk_size,
    	chunk_overlap  = chunk_size // 10,
	)
    return text_splitter.split_documents(doc)


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


if __name__ == '__main__':
    
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


    print('Loading Documents ...')
    # 定义文件路径
    pdf_file = "./data/2023级计算机科学与技术学术硕士研究生培养方案.pdf"
    md_file1 = "./data/历史沿革.md"
    md_file2 = "./data/学院简介.md"

    
    # 加载文件内容
    pdf_loader = PyPDFLoader(file_path=pdf_file)
    md_loader1 = UnstructuredMarkdownLoader(file_path=md_file1)
    md_loader2 = UnstructuredMarkdownLoader(file_path=md_file2)

    # 切分文档 - 根据实际段落情况进行切分
    pdf_documents = split(chunk_size=250, doc=pdf_loader.load()) # 培养方案段落更长
    md_documents1 = split(chunk_size=50, doc=md_loader1.load())
    md_documents2 = split(chunk_size=100, doc=md_loader2.load())
    documents = pdf_documents + md_documents1 + md_documents2

    # 日期加权
    for i in range(len(documents)):
        documents[i].page_content = weight_dates(documents[i].page_content)


    print('Loading Successfully!')
    print('Buiding Vector DB ...')
   
    # 调用词嵌入向量
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"]
    )

    # 构建数据库 使用FAISS
    db = FAISS.from_documents(documents, embeddings) 
    retriever = db.as_retriever( # 转换为检索器 返回最相关的k个文档
        search_kwargs = {
            'k': 7
    	}
    )

    print('Buiding Successfully!')
    # 调用模型
    llm = Tongyi( 
        # model_name="qwen1.5-1.8b-chat", 
        model_name="qwen2-72b-instruct",
        temperature=0.95, top_p=0.7, max_tokens=10, 
    )
    
    # 构建检索链
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 修改prompt 要求一句话回答问题 防止输出过多内容
    qa.combine_documents_chain.llm_chain.prompt.template = '''
Use the following pieces of context to answer the question directly at the end by a short sentence no more than 20 words.if you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Answer:
'''

    # 处理输入
    for i in range(num_questions):
        examples[i]['query'] = weight_dates(examples[i]['query'])

    
    # 测试链
    eval_chain = QAEvalChain.from_llm(llm)
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


    # 对测试问题进行回答
    print('Start To Query')
    predictions = qa.batch(inputs=examples)
    print('Query Completely')

    # print(predictions)

    # API冷却
    time.sleep(60)



    print('Start To Validation')
    graded_outputs = eval_chain.evaluate(examples, predictions)
    print('Result:')

    correct = 0

    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]["query"])
        print("Real Answer: " + predictions[i]["answer"])
        print("Predicted Answer: " + predictions[i]["result"])
        print("Predicted Grade: " + graded_outputs[i]["results"])
        print()

        if graded_outputs[i]["results"][:7].strip() == 'CORRECT': correct += 1

    print()
    print()
    print(f'Accuracy = {correct/num_questions * 100}%')



