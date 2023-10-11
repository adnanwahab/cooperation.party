#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install git+https://github.com/huggingface/transformers.git@refs/pull/25740/head accelerate')


# In[2]:


from transformers import AutoTokenizer
import transformers
import torch


# In[ ]:



from transformers import  LlamaForCausalLM, LlamaTokenizer, pipeline

base_model_path="./huggingface/llama7B"
model = LlamaForCausalLM.from_pretrained(base_model_path)
tokenizer =LlamaTokenizer.from_pretrained(base_model_path)


# In[5]:



#model = "codellama/CodeLlama-7b-Instruct-hf" #"codellama/CodeLlama-7b-hf"

get_ipython().system(' pip install optimum')
get_ipython().system(' pip install auto-gptq')


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
model_name_or_path = "TheBloke/CodeLlama-7B-Python-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)


# In[5]:



print(pipe('make a cool shader in webgpu')[0]['generated_text'])


# In[ ]:


pip install optimum` and auto-gptq library 'pip install auto-gptq'
import optimum 


# In[6]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install openai')
get_ipython().system('pip install langchain')
get_ipython().system('pip install sentence_transformers')
get_ipython().system('pip install requests')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install auto-gptq')
get_ipython().system('pip install peft==0.4.0')
get_ipython().system(' pip install -U git+https://github.com/huggingface/accelerate.git')

model_name_or_path = "TheBloke/CodeLlama-7B-GPTQ"
from transformers import AutoTokenizer, pipeline, logging

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,use_safetensors=True)


# In[ ]:


from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


model_name_or_path = "TheBloke/CodeLlama-7B-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,use_safetensors=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

embeddings=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small',model_kwargs={'device':'gpu:0'})


# In[7]:





# In[ ]:




