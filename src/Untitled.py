#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def makeNewFeatures():
    return 'make demo '
















#some group of 1000 people invented something
#this lets you vote with other people to use magic spells
#the spells take at least 3 people to collaborate and imagine together cool movies and so on
#all of movie club was on your side like a year ago


# In[7]:


import arxiv

# Search query
query = "algae photosynthesis"

# Search arXiv for the query
search = arxiv.Search(
  query=query,
  max_results=100,
)

# Loop through each paper found
for result in search.get():
    print("Title:", result.title)
    print("Authors:", result.authors)
    print("Summary:", result.summary)
    print("Link:", result.pdf_url)
    print("----")


# In[5]:


get_ipython().system('pip install arxiv')


# In[3]:


#from ipynb.fs.full.my_functions import getAllAirbnbInCityThatAreNotNoisy

import ipynb

from ipynb.fs.defs.geospatial import getAllAirbnbInCityThatAreNotNoisy

getAllAirbnbInCityThatAreNotNoisy()
#mental model for building application
#make 5 jupyter notebooks -> add like 100 functions that are cool
#in english notebook -> call jupyternotebook functions 
#in english notebook -> call javascript UI components which call jupyter notebook functions 
#TODO - use codellama to generate and add functions to javascript UI + jupyter notebook


# In[110]:


from rpc import crossReferenceAirbnb

crossReferenceAirbnb(1,2)


# In[ ]:


get all tweets that have pizza
order a pizza from instacart 
book flights to the pizzaria 
find all airbns that are close to the basketball stadium and have noise issues 


# In[ ]:


get_ipython().system('pip install --quiet bitsandbytes')
get_ipython().system('pip install --quiet transformers ')
get_ipython().system('pip install --quiet accelerate')
get_ipython().system('pip install scipy numpy')



import torch
print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path="./huggingface/llama7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16)

#Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True) 
tokenizer.save_pretrained(base_model_path, from_pt=True)


# In[69]:


json.load(open('encodings.json', 'r'))


# In[100]:


encodings = getEncodings(sentences)
str(encodings[0].tolist())


# In[88]:


##have to go downstirs
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import json 

sentences = ["get all tweets that mention pizza if geolocation",
 
 "get all football games next year",
 
 "order flights for each one",
 
 "get all twitch comments",
 
 "cluster twitch comments into groups",
]

def getPairs(documentOne, documentTwo):
    pairs = []
    cosine_scores = util.cos_sim(rhe[:664], pge)
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})  
#getPairs(encodings, )

def getEncodings(sentences):
    return model.encode(sentences, convert_to_tensor=True, device='cpu')

#keyed by encoding
def getProgram(sentence, program, type):
    encoding = getEncodings(sentence)
    key = str(encoding[0].tolist())
    program_generator_cache = json.load(open('encodings.json', 'r'))
    if encodings in program_generator_cache: return program_generator_cache[encodings]
    program_generator_cache[key] = program
    #if cannot find hand written version -> generate one using codeLlama
    #eventually -> decompose and generate smaller functions into larger ones that do what users want
    json.dump(program_generator_cache, open('encodings.json', 'w'))
    return {'fn': program_generator_cache[encodings]}

#get all airbnbs + cross reference with 311 - currently impossible without coding in python 
#write program by hand
#describe what it does
#send description to encoder 
#use encoding to fetch program from database 
#then you can use this to finetune the program.
[getProgram(sentence) for sentence in sentences]
#write file to disk


# In[87]:


open('./fetchTwitter.js').read()


# In[86]:


json.load(open('encodings.json', 'r'))


# In[61]:


from transformers import  LlamaForCausalLM, LlamaTokenizer, pipeline

base_model_path="./huggingface/llama7B"
model = LlamaForCausalLM.from_pretrained(base_model_path)
tokenizer =LlamaTokenizer.from_pretrained(base_model_path)


import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path="./huggingface/llama7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16)

#Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True) 
tokenizer.save_pretrained(base_model_path, from_pt=True)


# In[ ]:


model = "codellama/CodeLlama-7b-Instruct-hf" #"codellama/CodeLlama-7b-hf"

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                device_map="auto",
                max_new_tokens = 512,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
sequences = pipe(
    'I liked "Maneskin" and "Pink Floyd". Do you have any recommendations of other groups I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# In[ ]:


from transformers import AutoTokenizer
import transformers
import torch
tokenizer = AutoTokenizer.from_pretrained(model, token='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')
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


# In[ ]:


# set up inference via HuggingFace
from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-Instruct-hf" #"codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def sample_model(prompt):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1028,
    )
    return sequences[0]['generated_text']


# In[ ]:


from transformers import AutoTokenizer
import transformers
import torch

# Hugging face repo name
model = "codellama/CodeLlama-7b-Instruct-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

sequences = pipeline(
    'write the fibonacci program',
    do_sample=True,
    top_k=10,
    top_p = 0.9,
    temperature = 0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200, # can increase the length of sequence
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# In[ ]:


get_ipython().system('pip install huggingface_hub')

get_ipython().system('huggingface-cli login')


# In[ ]:


# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))


# In[ ]:


from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# In[ ]:



from transformers import AutoTokenizer, LlamaForCausalLM
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

# initialize the model

model_path = "Phind/Phind-CodeLlama-34B-v1"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# HumanEval helper

def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=256, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion

# perform HumanEval
problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)

# run `evaluate_functional_correctness samples.jsonl` in your HumanEval code sandbox


# In[ ]:


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="codellama/CodeLlama-34b-Instruct-hf")


# In[ ]:


from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# In[24]:


from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-13b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# In[ ]:


#solve singleplayer 

#1. travel planning
#2. collaboratively planning a garden with custom ferns - display an actual fasta, sgrna, pdbs and a ship to lab button based on an addres -> ask for address -> 200 lines of code away 
#3. literate programming for excel - write to an excel -> make a cool pattern like /r/place
#4. assist with building communities -> automate repetitive tasks ->
#5. assist with planning projects -> when code gets checked into github -> 
#6. teach people to code and have more fun learning math -> see the entire process can be used to do cool stuff -> trig -> hundred triangles becoming an origami crane
#7. [check] make a gant chart for cooking dinner -> snickerdoodles

0. make my robot clean my house every 3 days 
1. order these groceries every 3 days -> trader joes 


2. after an event -> write a 3 page article of everything you experienced and what questions you had 
-> write queries about text that transform the text into something that has an intersection with other peoples notes on the same event and then bundle and document everyones experience

today i went to a concert, it was cool. my favorite part was the 2nd song
my favorite part was the lights
half users favorite was lights, half was 2nd song and half was unknown.



#find intersection of two documents 

#encoding = 500 dimension that has no real meaning -> just used for statistical correlation within document 


# In[21]:


url = 'https://www.twitch.tv/cohhcarnage'

def getText(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to get the webpage")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    chat = soup.find_all('.chat-line__message')
    #copy(Array.from(document.querySelectorAll()).map(el => el.textContent))
    print(soup)
getText(url)



#make a 


# In[229]:


#! pip install requests beautifulsoup4

url = 'http://www.paulgraham.com/articles.html'

import requests
from bs4 import BeautifulSoup

def scrape_paul_graham_articles():
    url = "http://www.paulgraham.com/articles.html"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to get the webpage")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a'):
        link = a_tag.get('href', None)
        if link:
            links.append("http://www.paulgraham.com/" + link)
            
    return links

def getText(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to get the webpage")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    body = soup.find('body')
    
    if body:
        return body.get_text()


article_links = scrape_paul_graham_articles()

import json
with open('all_pg.txt', 'w') as file :
    json.dump([getText(link) for link in article_links], file)


# In[211]:


with open('./database.txt') as file:
    print(file.read())


# In[207]:


get_ipython().system('pip install openai')
get_ipython().system('pip install sentence-transformers')
get_ipython().system(' pip install datasets ')
get_ipython().system(' pip install WiktionaryParser')
get_ipython().system(' pip install transformers')
get_ipython().system('pip install faster-whisper')

get_ipython().system('pip install pronouncing')
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import openai
import time
import transformers
import os 
import whisper
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('universal_tagset')


# In[ ]:


videoplayback = '../data_sets/' + 'output.mp3'

# model = whisper.load_model("base")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio(videoplayback)
# #audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions(fp16=False)
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)
# from faster_whisper import WhisperModel

# model_size = "large-v2"

# # Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# # or run on GPU with INT8
# # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
# # model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe(videoplayback, beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#https://github.com/guillaumekln/faster-whisper


# In[ ]:


def getSimilarity(sentences):
    corpus_embeddings = model.encode(sentences, convert_to_tensor=True)
    #print(corpus_embeddings, sentences)
    clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.55)
    def process(item): return [sentences[i] for i in item]
    result = [process(item) for item in clusters ]
    # print(clusters)
    print('result,result,result',result)
    return result
random.shuffle(getSimilarity(stream_comments))
#if @ -> reply https://www.promptingguide.ai/tools


# In[ ]:


# lines = []
# with open('./archive/test_Arabic_tweets_negative_20190413.tsv') as f:
#     for line in f:
#         lines.append(line.split('\t')[1])

# openai.api_key = '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'
# len(lines)

# def translate_text(text, source_language, target_language):
#     print('translating ' + text)
#     prompt = f"Translate the following '{source_language}' text to '{target_language}': {text}"
#     first = time.perf_counter()
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that translates text."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=150,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#     second = time.perf_counter()
#     print(second - first)
#     translation = response.choices[0].message.content.strip()
#     return translation

# tweets = [translate_text(line, 'arabic', 'english') for line in lines[:10]]


# In[ ]:


def getPairs(documentOne, documentTwo):
    pairs = []
    cosine_scores = util.cos_sim(rhe[:664], pge)
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

#[ for tweet in tweets]


#reduce 3 paul graham essay and a gwern article into common subset of sentences with most similarity 
#LSTM -> looks at 4 sentences at a time -> tracks "meaning" of sentence as a 584 column vector and then 
#summarizies them -> and then finds similar parts 
#two notes have to be intersecting to be mergable 
#start with two notes that are mostly intersecting
#give everyone 90 minutes to write out everything they know about a topic - extemporaenous 
#merge notes 
#given n documents - reduce
    #given two documents -> 
    #return a list of clusters and their rank
        #rank each cluster by the relevance of each sentence to the title
        # map - return clusterID
        #within each cluster -> use NLTK to find the AST -> categorize as -> 'preposition, statement, question, jest, curiosity'
        # pairwise match most similar sentences - merge - TBD or out of scope
        
model = SentenceTransformer('all-MiniLM-L6-v2')
def encode(s1): return model.encode(s1, convert_to_tensor=True)


# In[ ]:


#find the right training dataset -> 300million reddit comments = 2 million downloads + research papers 
#what is the right training data set for "understanding twitch stream"

#sentiment analysis
#rank by humor
#rank by similarity 
#reorganize within topic
#assuming there are 5 topics
#person writes essay -> their essay is 'graded' or visualized as a venn diagram between theirs and nexus or their and all others 
#run spelling corrector uestionL how to find 300 non stream people to use app -> pay 300 students 5 dollars = 1500 
#find them a deal on 
#write an essay for 5 minutes 



#subject of sentence = non-person verb acted upon
#each classification has key structures that 

#theory -> i think
#mockery = save scumming ? 

#use sentiment to categorize 
#use feature vector to categorize
#use tags to categorize 
#see which one is more accurate
import random


tag_map = {
  "CC": "Coordinating conjunction",
  "CD": "Cardinal number",
  "DT": "Determiner",
  "EX": "Existential there",
  "FW": "Foreign word",
  "IN": "Preposition or subordinating conjunction",
  "JJ": "Adjective",
  "JJR": "Adjective, comparative",
  "JJS": "Adjective, superlative",
  "LS": "List item marker",
  "MD": "Modal",
  "NN": "Noun, singular or mass",
  "NNS": "Noun, plural",
  "NNP": "Proper noun, singular",
  "NNPS": "Proper noun, plural",
  "PDT": "Predeterminer",
  "POS": "Possessive ending",
  "PRP": "Personal pronoun",
  "PRP$": "Possessive pronoun",
  "RB": "Adverb",
  "RBR": "Adverb, comparative",
  "RBS": "Adverb, superlative",
  "RP": "Particle",
  "SYM": "Symbol",
  "TO": "to",
  "UH": "Interjection",
  "VB": "Verb, base form",
  "VBD": "Verb, past tense",
  "VBG": "Verb, gerund or present participle",
  "VBN": "Verb, past participle",
  "VBP": "Verb, non-3rd person singular present",
  "VBZ": "Verb, 3rd person singular present",
  "WDT": "Wh-determiner",
  "WP": "Wh-pronoun",
  "WP$": "Possessive wh-pronoun",
  "WRB": "Wh-adverb",
    ".": 'unknown_variable',
    ",": '',
    ':': '',
    '``': '',
    "''": ''
}
def getClassification(string):
    p = int(random.random() * 5)
    nouns = findNouns(string)
    verb_most_acted_on = nouns #findNouns(string)[0] if len(nouns) > 0 else ''
    return f'{classifications[p]}:  {" ".join(verb_most_acted_on)}'

def processTag(tagged_sentence):
    return [(orig,tag_map[actual_tag]) for (orig,actual_tag) in tagged_sentence if actual_tag in tag_map and 'noun']

def findNouns(string):
    return [noun for noun,tag in processTag(pos_tag(word_tokenize(string))) ]
#food 

#[findNouns(string) for string in stream_comments]
#within cluster -> tag topic by most common word if its that type of topic
#processTag(pos_tag(word_tokenize(stream_comments[0])))

[getClassification(string) for string in stream_comments]


# In[208]:


stream_comments = [
    "ZeroTepMusic: 12",
    "lysinehd: crabs in a bucket",
    "bouillabased: my life's biggest mistake was trusting my parents, especially when they told me education was the key to success.",
    "kylanc6: Americas is full of so many people with fucking peasant brain",
    "Replying to @spyderfrommars: btw, so many people have filed for bankruptcy and that used to mean having your debts forgiven. Probably many chatters parents have. SO MANY PEOPLE IN THE POLITICAL ARENA HAVE HAD THEIR DEBTS FORGIVEN. It's so stupid to use that arguementshainybug: and bankruptcy doesn't wipe student loans ",
    "MarIsMar: Corpa",
    "IsThatSalem: I hate this",
    "eronin37:   ",
    "v3sh_:  HYPERCLAP turn education into businesses",
    "hondewberry: CHATTERS NO, LET'S DO FUN SHIT BABY WHAT YOU DOIN",
    "JEZZ_7: Corpa",
    "xTrashPandaKingx: @HasanAbi you dont understand they're like one or two lucky breaks from being the 1%",
    "fearandrespect: Man that's a weird boot to be licking, chatter what the hell",
    "PrettyKrazy: profit motive destroys humanity",
    "Sutiibun_: ",
    "qwertyopsd2: Chatting really Hasan?",
    "Tamarama02: \"I couldn't care less.\"",
    "whitneythegoth: YEP LEECHES",
    "FriedWaffles:  just don't get sick or hurt",
    "redeyeink: gatekeeping and control over labor conditions",
    "BigDddyNick: sugar dads in chat?",
    "austrom: Corpa",
    "HalalChad_: America is a massive corporation",
    "russianspy619: Chatting I'm very smart",
    "dankusdingus: hasCapital",
    "dr_desu:  gimme gimme gimme",
    "LateAndNever: Pivo . o O ( Corpa 🔫  )",
    "phoneofff: @ashlynnicoleramirez report yourself as poor and never tell them anything ever again",
    "ADK_215: they want to make money of your student loan debt how do you guys not see this",
    "Leafy_Sh4de: Hey Hasan! My man! I know I longed for some twitch political commentary",
    "WhyYouGotNecklace: YEP just looking for more capital avenues",
    "meredyke: To profit in anyway imaginable",
    "imLunchy: Corpa private prisons",
    "thottopic666: YEP 󠀀",
    "thelookoutshift: that chatter didn't pay their loan and the govt took back their degree via lobotomy @hasanabi",
    "whataburgerfancyketchup: Knowledge is power Hasan. Its that simple. Keep people dumb, keep them powerless. @HasanAbi",
    "aquamiguel: Chatting 󠀀",
    "ya_plis: they need a controlled working force",
    "ShakeN_Bake: Fuck them",
    "moogerfooger_: can't get blood from a stone Crungo",
    "sandsim: making 17 year olds take loans out OMEGALUL",
    "stovetotheface: keep the masses dumb",
    "Resubscribe: @luckypompom qalla_s modCheck",
    "tr0piKEL1: I do that on medical bills as protest. I pay like $25/mo on $4,000 hospital bills  they don’t report it to credit as long as you pay.",
    "rentcontrolryan: just get a full ride scholarship EZ Chatting",
    "shoriu_: chat is so annoying today",
    "TheUh0hOreo:  HYPERCLAP",
    "Replying to @ashlynnicoleramirez: income driven repayment plan and stay poor forevershainybug: me lmao",
    "Shonnicus: paying a portion, you are still getting killed on the backend with interest. That's why they are fine with you not making full payments",
    "catboy_rai: housing, food, etc.",
    "YukiTsunoda__: ",
    "hashoe23: TATE",
    "Faviahn: Because they're told they have to pay more in taxes and have less money when they're already struggling.",
    "HerrosRevenge: ultimatley this argument boils down to \"its your fault for wanting to live and be happy\" most people dont want to go to school to work for the rest of their lives while paying to do so",
    "seeayy: almost every aspect of higher education is profitable",
    "TehAdamBomb: profit motives drive innocation ",
    "lordcharliesheen: YEE HAWWWW",
    "HUGEGAMER96: Military YEP",
    "clandestinie: Affordable but not always available",
    "big_dykeenergy: i’m below the income threshold so i get a payment of zero. i should clarify that i WORK FOR THE GOVERNMENT and i’m not paid enough to meet the thresholds",
    "aquamiguel: Corpa Clap",
    "bignachysosa: Healthcare doesnt have to be free but it SHOULDNT be private",
    "Cypres_warluckHyan8: americas kindaa fuked rn",
    "ZuzieZozo: I literally had a free operation",
    "quarantinewolf: Chatting Hasan @Hasanabi @Hasanthehun @Freedomeaglefuck",
    "IgiveBluebells: Educaiton is overpriced in the US",
    "librapelican: theres a reason my sociology class presented american exceptionalism as a form of propaganda",
    "xygeek: @hasanabi allow for bankruptcy, then normalize bankruptcy at graduation. Problem solved. KEKW",
    "Skill_Cylinder: YEP just join the military",
    "Hagasha: ",
    "bigstephfan: and here healthcare is so fucking expensive.",
    "BOATPARADE: can't have an educated proletariat",
    "RoguePr1nc355: Now it is about political ideology",
    "Zony66: you cant even have a fucking hobby without people asking you \"well how are you gunna make any money with that?\"",
    "cas3_: no war but class war",
    "Replying to @tr0piKEL1: I do that on medical bills as protest. I pay like $25/mo on $4,000 hospital bills rhyzKEK they don’t report it to credit as long as you pay.aspiration89: YEP 󠀀",
    "SimUser:  You want my number to not go up???",
     "narjuh: more than South Korea?",
    "punishedribcorn: Education isnt free for the same reason healthcare isnt free. Because you cant live without it @hasanabi",
    "GanjarDanks: @hasanabi true reason that education isn't free and student loans reign supreme is slabs",
    "ok_eevee:  Paywall the labor force @hasanabi",
    "kintu: there are worse neoliberal hellholes out there Aware",
    "esquerdomacho: I can get a free heart transplant in Brazil if I want MmmHmm",
    "lagsanaglasscoke: Corpa hehe",
    "ComradeCussy: Freedom ain't free brother @HasanAbi",
    "eronin37:   💵",
    "calimarx: It’s to maintain order",
    "PrettyKrazy: profit motive deprives every successful system",
    "SpanoNanoChano: even textbooks are a literal racket",
    "bignachysosa: Federalize it let the government deal with paying hospitals and doctors",
    "happppy_ant: YEP",
    "lardball1: @HasanAbi an educated proletariat is dynamite, like that reagan advisor said",
    "Replying to @ZuzieZozo: I literally had a free operationBurnzorr: You are one person",
    "PoogDoog: HE SAID THE THING LETSGO",
    "1337h4x: BALD POTATO PEELER OMEGALUL",
    "cms100210: All these things exist in countries hence it can work",
    "HVYHTTRS_: The biggest scam in college is the BOOKS, some good docs about it",
    "Eevee_Sprinkle:  Keep on licking the boot, GED Andy's.",
    "bakhtiari_veneco: Stupid question, is South Korea less capitalistic than America? @hasanabi",
    "JaychanLive:  WineTime PROFIT FIRST  WineTime",
    "whataburgerfancyketchup: Thats Me Pog",
    "dumpster27: message deleted by a moderator.",
    "moogerfooger_: like paying less than $15 min wage",
    "happppy_ant: YEP control",
    "Shroomie1707: Do you think you should be able to run for president if you are in jail @hasanabi",
    "DavidTheDaybed: D:",
    "Replying to @bignachysosa: Healthcare doesnt have to be free but it SHOULDNT be privateqwertyopsd2: it should be free",
    "Zpectr3: I pay like 300 Euros for university every semester in germany , but like 250 are for public transport. This shit is insane in the us @HasanAbi",
    "RamenBellic: @Baldpotatopeeler we just need to decommodify education.",
    "dicesettle: Lol. Don't do that",
    "mrbuddybuddy: KEKWait",
    "RowdyRoran: bro has been following for 3 years and is asking this now?",
    "sassoune: SORRY WE CAN BAIL OUT CMBS AT 30% purchase price - but when it comes to student loans we’re back to archaic - loan - predatory interest gurg payback or go die",
    "Darksoul9669: @hasanabi yeah man it was my own actions that had every part of my schooling telling me to take out loans and go to college as the only option and there being basically no downside. Really interesting how high school blows right through how devastating these loans were gonna be during these discussions when i was fucking 17 YEARS OLD",
    "Tetratera: university is free for everyone in argentina including foreigners, and you don't even have to take a standardized test, only have finished high school (and know upper intermediate spanish)",
    "lysinehd: permanent desperate underclass",
    "FALS3_g0D: crusing debt made to keep you a servant to the system",
    "atsign_: literally other countries can do it for free. is america not exceptional enough to do it?",
    "dumpster27: message deleted by a moderator.",
    "Fossabot: @dumpster27, Excessive spamming [warning]",
    "WeasleyLittleLiar: Did not used to cost that much",
    "rex__havoc: @hasanabi Have you talked about the new IDR plan \"SAVE\"? you're payments can be as low as 0/month",
    "thottopic666: every single aspect of this country was designed to suck the citizens dry as efficiently as possible",
    "c_d1999: Ask that chatter why don’t we charge for public high school!??",
    "thehappyparadox: YEP",
    "Skill_Cylinder: YEP",
    "kaimehra: yep",
    "kait516: YEP YEP YEP",
    "canola_oil: YEP",
    "thottopic666: YEP",
    "Hagasha: YEP"
]

stream_comments += [
    "MER_AKI: bro thinks hes him lol",
    "o7draco: ECO DEMON FRFR",
    "tko0_: UR SO LUCKY",
    "SparkYYY_123: SO LUCKY",
    "extratiarestrial: EWWWW",
    "tomas2brazy: Derke moment",
    "abhi_142: ECO king",
    "psygonnn: yeah yeah tarik we know you are going pro",
    "autumn0999: LOL",
    "betasimp42: Derke you was right Aware",
    "grandpafroggys: eco demon",
    "xDieWithPridex: whats his dpi and sens?",
    "lowertaxrates: KEKW ur insane sometimes",
    "PhanzGFX: A real one would get an ace there",
    "gangliaa: he predicted this",
    "MrKing8: KEKW",
    "AyoJabo: ECOOOO FRAGGGGGGER",
    "MandyLynx: calm down buddy",
    "Neon_Phaser: derke said it",
    "Fossabot: Hey, are you following tarik on Twitter? http://twitter.com/tarik",
    "rishon26: STOP OVERPEEKING LMFAO",
    "nopointgamer: eco frags",
    "demon_sl4: any cs2 news?",
    "aidenvovn420: overheat",
    "danielmacttv: You are him",
    "atinyspec: hallo",
    "GorillaTangie: KEKW",
    "ghost_khtab: KEKW",
    "丁乚仨乂 (tlex): KLİİPPPPPPPP",
    "oikawies: well ur consistent at overheating",
    "ZqCyzreN: ecobra",
    "ayoub_hh: ns",
    "bearrynice: @tarik you can satchel? Since when Lil bro",
    "KorHun_Official: kangkang gets 5 here @tarik",
    "suus001: OHHHH SHIT",
    "ub_zinio: overheaaat",
    "ironman_ap: sup ? @Derke",
    "Schabii97: DERKE W",
    "Grediann: overpeak = die Shruge",
    "jaybird1014: SIT DOWN PLS",
    "ayswoosh: @Derke how were champs?",
    "nishikoto: NASTY",
    "thickymonster: !duo",
    "wddcruz: 3King",
    "sqawg: Lil bro humbled himself",
    "Replying to @thickymonster: !duoFossabot: Asuna AYAYA",
    "littlesmchallowen: do that next round kekw",
    "davidakachuwy: COOKED then OVERPEEKED",
    "MER_AKI: you are not himothy",
    "lotace:     ",
    "ditt0o: we've got huge bets don't ROZA",
    "gme16: that spray transfer was lit as",
    "iicpr: overheat",
    "daymare5: it was horrible",
    "Sigfreed: NA BRAIN KEKW",
    "samsaraeyess: that spray transfer made me ink",
    "hwhevevsvb: no",
    "suus001: TUROK TUROK TUROK",
    "SilintNight: OMEGALUL",
    "Derke: NO",
    "mr_01ne: Derke knew it",
    "Replying to @Derke: i told uQuanFuPanda: deadass",
    "dioholic: terue",
    "ta3sk1: THIS TEAM IS FUCKING GOATED TARIK/STEW/ASUNA GGZ",
    "rue__s: heeey",
    "abcdgwenchana: overheat on eco",
    "AdderallBeforeBed: bet you can't do it again MmmHmm",
    "sissimou: fax",
    "xcrimsoncrookx: bro thinks the transfer was intentional AINTNOWAY",
    "adityasanas001: Ecodemon",
    "dioholic: true",
    "Lefluu: stew did everything there @tarik",
    "CosmicDeven: two eco frags and we start talking shit on derke KEKW",
    "Benjjamin: If you get 3 you're allowed to throw",
    "afor_f: its true",
    "Derke: IF ITS 5V1",
    "Harnasiek03: true",
    "lowertaxrates: no?",
    "ketosaiba11: replace jinggg no?",
    "theak44: BLABBERING BLABBERING",
    "Derke: AND I DIE FIRST",
    "laiiiny: You should apply for observer in VCT",
    "itsrawkus: wake up",
    "tripharder: ahh yes the rule",
    "shruggy8: TRUEING",
    "autumn0999: nice fucking shots tho",
    "Derke: ITS MY FAULT",
    "rishon26: @Derke get this man on fnatic",
    "OzGunAim: !sens",
    "Fossabot: CSGO: 1.5 @ 800 DPI, VALORANT: .471 800 DPI",
    "JRD_Nath: ",
    "xkillo147: True, NA rule",
    "gentlecpu: KEKW if you get 1 it's not your fault",
    "jinsoooo: if you get 2 you go for the ace",
    "rightylucy: Lkekw",
    "lionbrav3: C9 VIBEZ",
    "emil__val: KEKW KEKW",
    "maareeyyyy: !mouse",
    "FarmerFelox: In NA if you get 1 go for 5",
    "abcdgwenchana: eco frag",
    "alirezathe1: !res",
    "Fossabot: DeathAdder V3 Pro",
    "Fossabot: Val 16:10 (1680x1050) - CSGO: 1280x960",
    "CaliKillz3: TRUEING",
    "rentr04: homie turned up cuz derke is watching. respect",
    "hyp3r10n2: @tarik gets 3 wins round then overfaces and gets mad for it xD",
    "rightylucy: KEKW",
    "riyuoh: IF U GET 3 YOU CAN OVERHEAT 100%",
    "PiquesGaming: thats facts tho",
    "gkhn94: Dayi bi kere turkce konus be",
    "Sigfreed: LOOK ITS A 1V1 NOW",
    "丁乚仨乂 (tlex): KLİPPP",
    "myinnerfaye: Himothy is that you?",
    "Maximus6267: KEKW no way",
    "tsylogy: @Derke 5V1 DSG Aware",
    "ub_zinio: derkes fault",
    "shruggy8: gonna lose PepeLaugh",
    "Sigfreed: ITS A FUCKING 1V1 NOW",
    "Replying to @lionbrav3: C9 VIBEZXeppaa: ?",
    "ItsTavyy: maybe you need to peek more @tarik",
    "danielmacttv: Its DERKE’s fault",
    "dexterityCS: KEKW 󠀀",
    "Rickz10K: KEKW KEKW KEKW KEKW",
    "kaizo_rm: HUH 󠀀",
    "slaxxxyyyy: Fair enuff",
    "diipsy9: AYOO HH",
    "h1k1k0_: HUH",
    "m0gi08: !gekko",
    "Apollo_Neptune: HUH",
    "Fossabot: LilBro it's gekkin time ezz",
    "siwa33: Close gamba mods",
    "wahbi_79: HUH",
    "xelzttv: HUH",
    "emil__val: KEKW",
    "ig5mindhacker: HUH",
    "krasqu33: HUH",
    "aqilus: HUH",
    "Jordbaermelk: WOT",
    "derkesdoormat: @derke OOO DERKE'S HERE HII",
    "beepbopp11: HUH",
    "cenk4k: HUH",
    "mrsteallyourcat: HUH",
    "xclaassic: true",
    "Aethielle: HUH",
    "cyb_eric: HUH",
    "shruggy8: Sadge",
    "mesme_R: HUH"
]
stream_comments = [comment.split(':')[1] for comment in stream_comments]
stream_comments = [comment for comment in stream_comments if len(comment.strip()) > 0]
    
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

#You can then use this pipeline to classify sequences into any of the class names you specify.
classifications = """Retraction
Explanation
Query
Suggestion
Ammendment
Expletive
Answer
Recitation
stament 
observation
Commentary
Conclusion
Mockery
Qualification 
Objection
Theory
Continuation
Evaluation
Clarification
""".split('\n')
sequence_to_classify = "one day I will see the world"
candidate_labels = classifications
classifier(stream_comments[0], candidate_labels)
# with open('./stream-comments.json','r+') as file:
#       # First we load existing data into a dict.
#     file_data = json.load(file)
#     # Join new_data with file_data inside emp_details
#     file_data += stream_comments
#     # Sets file's current position at offset.
#     file.seek(0)
#     # convert back to json.
#     json.dump(file_data, file, indent = 4)
 
#copy(Array.from(document.querySelectorAll('.chat-line__message')).map(el => el.textContent))

# from transformers import pipeline
# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
# sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"


# In[210]:


import torch
from transformers import BartForConditionalGeneration, BartTokenizer

input_sentence = "there are 300 cats in the neighborhood ."

model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
batch = tokenizer(input_sentence, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_sentence)


# In[22]:


classifier(stream_comments[0], classifications, batch_size=8)


# In[ ]:


for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass


# In[28]:


import time

one = time.time()
isClassified = [classifier(seq, classifications, multi_label=False, batch_size=256) for seq in stream_comments[:20]]
two = time.time()
print(two - one)
#260


# In[16]:


isClassified

for i in isClassified:
    i['labels'] = [s for index, s in enumerate(i['labels']) if i['scores'][index] > .5]
    i['scores'] = [s for index, s in enumerate(i['scores']) if s > .5]

            
isClassified


# In[1]:


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)


# In[5]:


(input["input_ids"].to(device))


# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device('cuda:0')
#torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
hypothesis = "Emmanuel Macron is the President of France"

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to('cpu'))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)


# In[2]:





# In[3]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
stream_comments = [
    "its_pam_ela: That chatter was wrong you can’t just pay something",
    "ZeroTepMusic: 12",
    "lysinehd: crabs in a bucket",
    "bouillabased: my life's biggest mistake was trusting my parents, especially when they told me education was the key to success.",
    "kylanc6: Americas is full of so many people with fucking peasant brain",
    "Replying to @spyderfrommars: btw, so many people have filed for bankruptcy and that used to mean having your debts forgiven. Probably many chatters parents have. SO MANY PEOPLE IN THE POLITICAL ARENA HAVE HAD THEIR DEBTS FORGIVEN. It's so stupid to use that arguementshainybug: and bankruptcy doesn't wipe student loans ",
    "MarIsMar: Corpa",
    "IsThatSalem: I hate this",
    "eronin37:   ",
    "v3sh_:  HYPERCLAP turn education into businesses",
    "hondewberry: CHATTERS NO, LET'S DO FUN SHIT BABY WHAT YOU DOIN",
    "JEZZ_7: Corpa",
    "xTrashPandaKingx: @HasanAbi you dont understand they're like one or two lucky breaks from being the 1%",
    "fearandrespect: Man that's a weird boot to be licking, chatter what the hell",
    "PrettyKrazy: profit motive destroys humanity",
    "Sutiibun_: ",
    "qwertyopsd2: Chatting really Hasan?",
    "Tamarama02: \"I couldn't care less.\"",
    "whitneythegoth: YEP LEECHES",
    "FriedWaffles:  just don't get sick or hurt",
    "redeyeink: gatekeeping and control over labor conditions",
    "BigDddyNick: sugar dads in chat?",
    "austrom: Corpa",
    "HalalChad_: America is a massive corporation",
    "russianspy619: Chatting I'm very smart",
    "dankusdingus: hasCapital",
    "dr_desu:  gimme gimme gimme",
    "LateAndNever: Pivo . o O ( Corpa 🔫  )",
    "phoneofff: @ashlynnicoleramirez report yourself as poor and never tell them anything ever again",
    "ADK_215: they want to make money of your student loan debt how do you guys not see this",
    "Leafy_Sh4de: Hey Hasan! My man! I know I longed for some twitch political commentary",
    "WhyYouGotNecklace: YEP just looking for more capital avenues",
    "meredyke: To profit in anyway imaginable",
    "imLunchy: Corpa private prisons",
    "thottopic666: YEP 󠀀",
    "thelookoutshift: that chatter didn't pay their loan and the govt took back their degree via lobotomy @hasanabi",
    "whataburgerfancyketchup: Knowledge is power Hasan. Its that simple. Keep people dumb, keep them powerless. @HasanAbi",
    "aquamiguel: Chatting 󠀀",
    "ya_plis: they need a controlled working force",
    "ShakeN_Bake: Fuck them",
    "moogerfooger_: can't get blood from a stone Crungo",
    "sandsim: making 17 year olds take loans out OMEGALUL",
    "stovetotheface: keep the masses dumb",
    "Resubscribe: @luckypompom qalla_s modCheck",
    "tr0piKEL1: I do that on medical bills as protest. I pay like $25/mo on $4,000 hospital bills  they don’t report it to credit as long as you pay.",
    "rentcontrolryan: just get a full ride scholarship EZ Chatting",
    "shoriu_: chat is so annoying today",
    "TheUh0hOreo:  HYPERCLAP",
    "Replying to @ashlynnicoleramirez: income driven repayment plan and stay poor forevershainybug: me lmao",
    "Shonnicus: paying a portion, you are still getting killed on the backend with interest. That's why they are fine with you not making full payments",
    "catboy_rai: housing, food, etc.",
    "YukiTsunoda__: ",
    "hashoe23: TATE",
    "Faviahn: Because they're told they have to pay more in taxes and have less money when they're already struggling.",
    "duskdeserter: this country fucken sucks",
    "HerrosRevenge: ultimatley this argument boils down to \"its your fault for wanting to live and be happy\" most people dont want to go to school to work for the rest of their lives while paying to do so",
    "seeayy: almost every aspect of higher education is profitable",
    "chelseymakes: Daddy chill 🫠",
    "TehAdamBomb: profit motives drive innocation ",
    "lordcharliesheen: YEE HAWWWW",
    "HUGEGAMER96: Military YEP",
    "clandestinie: Affordable but not always available",
    "big_dykeenergy: i’m below the income threshold so i get a payment of zero. i should clarify that i WORK FOR THE GOVERNMENT and i’m not paid enough to meet the thresholds",
    "aquamiguel: Corpa Clap",
    "bignachysosa: Healthcare doesnt have to be free but it SHOULDNT be private",
    "Cypres_warluckHyan8: americas kindaa fuked rn",
    "ZuzieZozo: I literally had a free operation",
    "quarantinewolf: Chatting Hasan @Hasanabi @Hasanthehun @Freedomeaglefuck",
    "IgiveBluebells: Educaiton is overpriced in the US",
    "librapelican: theres a reason my sociology class presented american exceptionalism as a form of propaganda",
    "xygeek: @hasanabi allow for bankruptcy, then normalize bankruptcy at graduation. Problem solved. KEKW",
    "Skill_Cylinder: YEP just join the military",
    "Hagasha: ",
    "bigstephfan: and here healthcare is so fucking expensive.",
    "BOATPARADE: can't have an educated proletariat",
    "RoguePr1nc355: Now it is about political ideology",
    "Zony66: you cant even have a fucking hobby without people asking you \"well how are you gunna make any money with that?\"",
    "cas3_: no war but class war",
    "Replying to @tr0piKEL1: I do that on medical bills as protest. I pay like $25/mo on $4,000 hospital bills rhyzKEK they don’t report it to credit as long as you pay.aspiration89: YEP 󠀀",
    "SimUser:  You want my number to not go up???",
    "politicsenjoyer:  dumb consumer slaves",
    "narjuh: more than South Korea?",
    "punishedribcorn: Education isnt free for the same reason healthcare isnt free. Because you cant live without it @hasanabi",
    "GanjarDanks: @hasanabi true reason that education isn't free and student loans reign supreme is slabs",
    "ok_eevee:  Paywall the labor force @hasanabi",
    "kintu: there are worse neoliberal hellholes out there Aware",
    "esquerdomacho: I can get a free heart transplant in Brazil if I want MmmHmm",
    "lagsanaglasscoke: Corpa hehe",
    "ComradeCussy: Freedom ain't free brother @HasanAbi",
    "eronin37:   💵",
    "calimarx: It’s to maintain order",
    "PrettyKrazy: profit motive deprives every successful system",
    "SpanoNanoChano: even textbooks are a literal racket",
    "bignachysosa: Federalize it let the government deal with paying hospitals and doctors",
    "happppy_ant: YEP",
    "lardball1: @HasanAbi an educated proletariat is dynamite, like that reagan advisor said",
    "Replying to @ZuzieZozo: I literally had a free operationBurnzorr: You are one person",
    "PoogDoog: HE SAID THE THING LETSGO",
    "1337h4x: BALD POTATO PEELER OMEGALUL",
    "cms100210: All these things exist in countries hence it can work",
    "HVYHTTRS_: The biggest scam in college is the BOOKS, some good docs about it",
    "Eevee_Sprinkle:  Keep on licking the boot, GED Andy's.",
    "bakhtiari_veneco: Stupid question, is South Korea less capitalistic than America? @hasanabi",
    "JaychanLive:  WineTime PROFIT FIRST  WineTime",
    "whataburgerfancyketchup: Thats Me Pog",
    "dumpster27: message deleted by a moderator.",
    "moogerfooger_: like paying less than $15 min wage",
    "happppy_ant: YEP control",
    "Shroomie1707: Do you think you should be able to run for president if you are in jail @hasanabi",
    "DavidTheDaybed: D:",
    "Replying to @bignachysosa: Healthcare doesnt have to be free but it SHOULDNT be privateqwertyopsd2: it should be free",
    "Zpectr3: I pay like 300 Euros for university every semester in germany , but like 250 are for public transport. This shit is insane in the us @HasanAbi",
    "RamenBellic: @Baldpotatopeeler we just need to decommodify education.",
    "dicesettle: Lol. Don't do that",
    "mrbuddybuddy: KEKWait",
    "RowdyRoran: bro has been following for 3 years and is asking this now?",
    "sassoune: SORRY WE CAN BAIL OUT CMBS AT 30% purchase price - but when it comes to student loans we’re back to archaic - loan - predatory interest gurg payback or go die",
    "Darksoul9669: @hasanabi yeah man it was my own actions that had every part of my schooling telling me to take out loans and go to college as the only option and there being basically no downside. Really interesting how high school blows right through how devastating these loans were gonna be during these discussions when i was fucking 17 YEARS OLD",
    "Tetratera: university is free for everyone in argentina including foreigners, and you don't even have to take a standardized test, only have finished high school (and know upper intermediate spanish)",
    "lysinehd: permanent desperate underclass",
    "FALS3_g0D: crusing debt made to keep you a servant to the system",
    "sandsim: literally scamming children",
    "atsign_: literally other countries can do it for free. is america not exceptional enough to do it?",
    "dumpster27: message deleted by a moderator.",
    "Fossabot: @dumpster27, Excessive spamming [warning]",
    "WeasleyLittleLiar: Did not used to cost that much",
    "rex__havoc: @hasanabi Have you talked about the new IDR plan \"SAVE\"? you're payments can be as low as 0/month",
    "thottopic666: every single aspect of this country was designed to suck the citizens dry as efficiently as possible",
    "c_d1999: Ask that chatter why don’t we charge for public high school!??",
    "thehappyparadox: YEP",
    "Skill_Cylinder: YEP",
    "kaimehra: yep",
    "kait516: YEP YEP YEP",
    "canola_oil: YEP",
    "thottopic666: YEP",
    "Hagasha: YEP"
]

stream_comments += [
    "MER_AKI: bro thinks hes him lol",
    "xmas31: That spray so mad u really him",
    "o7draco: ECO DEMON FRFR",
    "tko0_: UR SO LUCKY",
    "SparkYYY_123: SO LUCKY",
    "extratiarestrial: EWWWW",
    "tomas2brazy: Derke moment",
    "abhi_142: ECO king",
    "psygonnn: yeah yeah tarik we know you are going pro",
    "autumn0999: LOL",
    "betasimp42: Derke you was right Aware",
    "grandpafroggys: eco demon",
    "xDieWithPridex: whats his dpi and sens?",
    "lowertaxrates: KEKW ur insane sometimes",
    "PhanzGFX: A real one would get an ace there",
    "gangliaa: he predicted this",
    "MrKing8: KEKW",
    "AyoJabo: ECOOOO FRAGGGGGGER",
    "MandyLynx: calm down buddy",
    "Neon_Phaser: derke said it",
    "Fossabot: Hey, are you following tarik on Twitter? http://twitter.com/tarik",
    "rishon26: STOP OVERPEEKING LMFAO",
    "nopointgamer: eco frags",
    "demon_sl4: any cs2 news?",
    "aidenvovn420: overheat",
    "danielmacttv: You are him",
    "atinyspec: hallo",
    "GorillaTangie: KEKW",
    "ghost_khtab: KEKW",
    "丁乚仨乂 (tlex): KLİİPPPPPPPP",
    "oikawies: well ur consistent at overheating",
    "ZqCyzreN: ecobra",
    "ayoub_hh: ns",
    "bearrynice: @tarik you can satchel? Since when Lil bro",
    "KorHun_Official: kangkang gets 5 here @tarik",
    "suus001: OHHHH SHIT",
    "ub_zinio: overheaaat",
    "ironman_ap: sup ? @Derke",
    "Schabii97: DERKE W",
    "Grediann: overpeak = die Shruge",
    "jaybird1014: SIT DOWN PLS",
    "ayswoosh: @Derke how were champs?",
    "nishikoto: NASTY",
    "thickymonster: !duo",
    "wddcruz: 3King",
    "sqawg: Lil bro humbled himself",
    "Replying to @thickymonster: !duoFossabot: Asuna AYAYA",
    "littlesmchallowen: do that next round kekw",
    "davidakachuwy: COOKED then OVERPEEKED",
    "MER_AKI: you are not himothy",
    "lotace:     ",
    "ditt0o: we've got huge bets don't ROZA",
    "gme16: that spray transfer was lit as",
    "iicpr: overheat",
    "daymare5: it was horrible",
    "Sigfreed: NA BRAIN KEKW",
    "samsaraeyess: that spray transfer made me ink",
    "hwhevevsvb: no",
    "suus001: TUROK TUROK TUROK",
    "SilintNight: OMEGALUL",
    "Derke: NO",
    "mr_01ne: Derke knew it",
    "Replying to @Derke: i told uQuanFuPanda: deadass",
    "dioholic: terue",
    "ta3sk1: THIS TEAM IS FUCKING GOATED TARIK/STEW/ASUNA GGZ",
    "rue__s: heeey",
    "abcdgwenchana: overheat on eco",
    "AdderallBeforeBed: bet you can't do it again MmmHmm",
    "sissimou: fax",
    "xcrimsoncrookx: bro thinks the transfer was intentional AINTNOWAY",
    "adityasanas001: Ecodemon",
    "dioholic: true",
    "Lefluu: stew did everything there @tarik",
    "CosmicDeven: two eco frags and we start talking shit on derke KEKW",
    "Benjjamin: If you get 3 you're allowed to throw",
    "afor_f: its true",
    "Derke: IF ITS 5V1",
    "Harnasiek03: true",
    "lowertaxrates: no?",
    "ketosaiba11: replace jinggg no?",
    "theak44: BLABBERING BLABBERING",
    "Derke: AND I DIE FIRST",
    "laiiiny: You should apply for observer in VCT",
    "itsrawkus: wake up",
    "tripharder: ahh yes the rule",
    "shruggy8: TRUEING",
    "autumn0999: nice fucking shots tho",
    "Derke: ITS MY FAULT",
    "rishon26: @Derke get this man on fnatic",
    "OzGunAim: !sens",
    "Fossabot: CSGO: 1.5 @ 800 DPI, VALORANT: .471 800 DPI",
    "JRD_Nath: ",
    "xdpotatolord: @tarik UR BICEPS ARE HUGE!!!",
    "xkillo147: True, NA rule",
    "gentlecpu: KEKW if you get 1 it's not your fault",
    "jinsoooo: if you get 2 you go for the ace",
    "rightylucy: Lkekw",
    "lionbrav3: C9 VIBEZ",
    "emil__val: KEKW KEKW",
    "maareeyyyy: !mouse",
    "FarmerFelox: In NA if you get 1 go for 5",
    "abcdgwenchana: eco frag",
    "alirezathe1: !res",
    "Fossabot: DeathAdder V3 Pro",
    "Fossabot: Val 16:10 (1680x1050) - CSGO: 1280x960",
    "CaliKillz3: TRUEING",
    "rentr04: homie turned up cuz derke is watching. respect",
    "hyp3r10n2: @tarik gets 3 wins round then overfaces and gets mad for it xD",
    "rightylucy: KEKW",
    "riyuoh: IF U GET 3 YOU CAN OVERHEAT 100%",
    "PiquesGaming: thats facts tho",
    "gkhn94: Dayi bi kere turkce konus be",
    "Sigfreed: LOOK ITS A 1V1 NOW",
    "丁乚仨乂 (tlex): KLİPPP",
    "myinnerfaye: Himothy is that you?",
    "Maximus6267: KEKW no way",
    "tsylogy: @Derke 5V1 DSG Aware",
    "ub_zinio: derkes fault",
    "itzzero3: arabic blood",
    "shruggy8: gonna lose PepeLaugh",
    "Sigfreed: ITS A FUCKING 1V1 NOW",
    "Replying to @lionbrav3: C9 VIBEZXeppaa: ?",
    "ItsTavyy: maybe you need to peek more @tarik",
    "danielmacttv: Its DERKE’s fault",
    "dexterityCS: KEKW 󠀀",
    "Rickz10K: KEKW KEKW KEKW KEKW",
    "kaizo_rm: HUH 󠀀",
    "slaxxxyyyy: Fair enuff",
    "diipsy9: AYOO HH",
    "h1k1k0_: HUH",
    "m0gi08: !gekko",
    "Apollo_Neptune: HUH",
    "Fossabot: LilBro it's gekkin time ezz",
    "siwa33: Close gamba mods",
    "wahbi_79: HUH",
    "xelzttv: HUH",
    "emil__val: KEKW",
    "ig5mindhacker: HUH",
    "krasqu33: HUH",
    "aqilus: HUH",
    "Jordbaermelk: WOT",
    "derkesdoormat: @derke OOO DERKE'S HERE HII",
    "beepbopp11: HUH",
    "cenk4k: HUH",
    "mrsteallyourcat: HUH",
    "xclaassic: true",
    "Aethielle: HUH",
    "cyb_eric: HUH",
    "shruggy8: Sadge",
    "mesme_R: HUH"
]
stream_comments = [comment.split(':')[1] for comment in stream_comments]
stream_comments = [comment for comment in stream_comments if len(comment.strip()) > 0]
    
premise = stream_comments[:10]
label = 'cool'
hypothesis = f'This example is {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
logits = nli_model(x.to('cpu'))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]


# In[ ]:


prob_label_is_true


# In[ ]:


from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
(summarizer(stream_comments, max_length=20, min_length=5, do_sample=False))


# In[2]:





# In[ ]:


from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
sentiments = sentiment_pipeline(stream_comments)

[f"{stream_comments[i]}   {sentiments[i]['label']}  {sentiments[i]['score']}" for i, char in enumerate(sentiments)]


# In[ ]:


+from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
inputs = tokenizer("eat lots of green beans and black eyed ", return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

[tokenizer.decode(string, skip_special_tokens=True) for string in outputs]


#get summary of them
#organize by cluster
    #sentiment
    
    
#download clips https://youtu.be/HigmUsGEEww -> get transcript to provide tagging
#download https://www.youtube.com/watch?v=HigmUsGEEww&feature=youtu.be&ab_channel=Joe-Astro


# In[ ]:


billsum = load_dataset("billsum", split="ca_test")


# In[ ]:


from transformers import AutoTokenizer
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# In[ ]:


prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:





#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)


# In[ ]:


from transformers import pipeline
import os

## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
text = """One month after the United States began what has become a troubled rollout of a national COVID vaccination campaign, the effort is finally gathering real steam.
Close to a million doses -- over 951,000, to be more exact -- made their way into the arms of Americans in the past 24 hours, the U.S. Centers for Disease Control and Prevention reported Wednesday. That's the largest number of shots given in one day since the rollout began and a big jump from the previous day, when just under 340,000 doses were given, CBS News reported.
That number is likely to jump quickly after the federal government on Tuesday gave states the OK to vaccinate anyone over 65 and said it would release all the doses of vaccine it has available for distribution. Meanwhile, a number of states have now opened mass vaccination sites in an effort to get larger numbers of people inoculated, CBS News reported."""
#Summarize
summary_text = summarizer('\n'.join(stream_comments)[:1022], max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)


# In[ ]:



def getClusters(sentences):
    clusters = util.community_detection(encode(sentences), min_community_size=1, threshold=0.5)
    clusters

    def process(item): return [sentences[i] for i in item]
    result = [process(item) for item in clusters ]
    return result

clusters = getClusters(stream_comments)


# In[ ]:





# In[ ]:


from wiktionaryparser import WiktionaryParser

parser = WiktionaryParser()
word = parser.fetch('eating')
#another_word = parser.fetch('test', 'french')
# parser.set_default_language('french')
# parser.exclude_part_of_speech('noun')
# parser.include_relation('alternative forms')
word


# In[ ]:


import gensim.downloader as api

# Load the pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

def generalize_word(word):
    try:
        similar_words = w2v_model.most_similar(positive=[word], topn=5)
        generalized_word = similar_words[0][0]  # Get the most similar word
        return generalized_word
    except KeyError:
        return None

word = "cake"
generalized_word = generalize_word(word)
if generalized_word:
    print(f"The generalized term for '{word}' is '{generalized_word}'.")
else:
    print(f"No generalization found for '{word}'.")

word = "burger"
generalized_word = generalize_word(word)
if generalized_word:
    print(f"The generalized term for '{word}' is '{generalized_word}'.")
else:
    print(f"No generalization found for '{word}'.")


# In[ ]:


get_ipython().system(' pip install gensim')


# In[ ]:


#categorize heckling

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def generalize_word(word):
    synsets = wordnet.synsets(word)
    
    if not synsets:
        return None
    
    # Get the first synset (most common meaning)
    synset = synsets[0]
    
    # Find hypernyms (more general terms)
    hypernyms = synset.hypernyms()
    
    if not hypernyms:
        return None
    
    # Get the first hypernym (more general term)
    hypernym = hypernyms[0]
    
    # Extract the lemma name of the hypernym
    generalized_word = hypernym.lemmas()[0].name()
    
    return generalized_word

word = "toast"
generalized_word = generalize_word(word)
if generalized_word:
    print(f"The generalized term for '{word}' is '{generalized_word}'.")
else:
    print(f"No generalization found for '{word}'.")

word = "hamburger"
generalized_word = generalize_word(word)
if generalized_word:
    print(f"The generalized term for '{word}' is '{generalized_word}'.")
else:
    print(f"No generalization found for '{word}'.")
    
    
t = wordnet.synsets('popcorn')


# In[ ]:





# In[ ]:


def getTopicTitle(text):
    print('translating ' + text)
    prompt = f"How would you classify '{text}' as a topic?"
    first = time.perf_counter()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    second = time.perf_counter()
    print(second - first)
    translation = response.choices[0].message.content.strip()
    return translation

getTopicTitle(stream_comments[10])

stream_comments[10]


# In[ ]:



rh= """Talk at Bellcore, 7 March 1986

The title of my talk is "You and Your Research." It is not about managing research, it is about how you individually do your research. I could give a talk on the other subject — but it's not, it's about you. I'm not talking about ordinary run-of-the-mill research; I'm talking about great research. And for the sake of describing great research I'll occasionally say Nobel-Prize type of work. It doesn't have to gain the Nobel Prize, but I mean those kinds of things which we perceive are significant things. Relativity, if you want, Shannon's information theory, any number of outstanding theories — that's the kind of thing I'm talking about.

Now, how did I come to do this study? At Los Alamos I was brought in to run the computing machines which other people had got going, so those scientists and physicists could get back to business. I saw I was a stooge. I saw that although physically I was the same, they were different. And to put the thing bluntly, I was envious. I wanted to know why they were so different from me. I saw Feynman up close. I saw Fermi and Teller. I saw Oppenheimer. I saw Hans Bethe: he was my boss. I saw quite a few very capable people. I became very interested in the difference between those who do and those who might have done.

When I came to Bell Labs, I came into a very productive department. Bode was the department head at the time; Shannon was there, and there were other people. I continued examining the questions, "Why?" and "What is the difference?" I continued subsequently by reading biographies, autobiographies, asking people questions such as: "How did you come to do this?" I tried to find out what are the differences. And that's what this talk is about.

Now, why is this talk important? I think it is important because, as far as I know, each of you has one life to live. Even if you believe in reincarnation it doesn't do you any good from one life to the next! Why shouldn't you do significant things in this one life, however you define significant? I'm not going to define it — you know what I mean. I will talk mainly about science because that is what I have studied. But so far as I know, and I've been told by others, much of what I say applies to many fields. Outstanding work is characterized very much the same way in most fields, but I will confine myself to science.

In order to get at you individually, I must talk in the first person. I have to get you to drop modesty and say to yourself, "Yes, I would like to do first-class work." Our society frowns on people who set out to do really good work. You're not supposed to; luck is supposed to descend on you and you do great things by chance. Well, that's a kind of dumb thing to say. I say, why shouldn't you set out to do something significant. You don't have to tell other people, but shouldn't you say to yourself, "Yes, I would like to do something significant."

In order to get to the second stage, I have to drop modesty and talk in the first person about what I've seen, what I've done, and what I've heard. I'm going to talk about people, some of whom you know, and I trust that when we leave, you won't quote me as saying some of the things I said.

Let me start not logically, but psychologically. I find that the major objection is that people think great science is done by luck. It's all a matter of luck. Well, consider Einstein. Note how many different things he did that were good. Was it all luck? Wasn't it a little too repetitive? Consider Shannon. He didn't do just information theory. Several years before, he did some other good things and some which are still locked up in the security of cryptography. He did many good things.

You see again and again that it is more than one thing from a good person. Once in a while a person does only one thing in his whole life, and we'll talk about that later, but a lot of times there is repetition. I claim that luck will not cover everything. And I will cite Pasteur who said, "Luck favors the prepared mind." And I think that says it the way I believe it. There is indeed an element of luck, and no, there isn't. The prepared mind sooner or later finds something important and does it. So yes, it is luck. The particular thing you do is luck, but that you do something is not.

For example, when I came to Bell Labs, I shared an office for a while with Shannon. At the same time he was doing information theory, I was doing coding theory. It is suspicious that the two of us did it at the same place and at the same time — it was in the atmosphere. And you can say, "Yes, it was luck." On the other hand you can say, "But why of all the people in Bell Labs then were those the two who did it?" Yes, it is partly luck, and partly it is the prepared mind; but "partly" is the other thing I'm going to talk about. So, although I'll come back several more times to luck, I want to dispose of this matter of luck as being the sole criterion whether you do great work or not. I claim you have some, but not total, control over it. And I will quote, finally, Newton on the matter. Newton said, "If others would think as hard as I did, then they would get similar results."

One of the characteristics you see, and many people have it including great scientists, is that usually when they were young they had independent thoughts and had the courage to pursue them. For example, Einstein, somewhere around 12 or 14, asked himself the question, "What would a light wave look like if I went with the velocity of light to look at it?" Now he knew that electromagnetic theory says you cannot have a stationary local maximum. But if he moved along with the velocity of light, he would see a local maximum. He could see a contradiction at the age of 12, 14, or somewhere around there, that everything was not right and that the velocity of light had something peculiar. Is it luck that he finally created special relativity? Early on, he had laid down some of the pieces by thinking of the fragments. Now that's the necessary but not sufficient condition. All of these items I will talk about are both luck and not luck.

How about having lots of brains? It sounds good. Most of you in this room probably have more than enough brains to do first-class work. But great work is something else than mere brains. Brains are measured in various ways. In mathematics, theoretical physics, astrophysics, typically brains correlates to a great extent with the ability to manipulate symbols. And so the typical IQ test is apt to score them fairly high. On the other hand, in other fields it is something different. For example, Bill Pfann, the fellow who did zone melting, came into my office one day. He had this idea dimly in his mind about what he wanted and he had some equations. It was pretty clear to me that this man didn't know much mathematics and he wasn't really articulate. His problem seemed interesting so I took it home and did a little work. I finally showed him how to run computers so he could compute his own answers. I gave him the power to compute. He went ahead, with negligible recognition from his own department, but ultimately he has collected all the prizes in the field. Once he got well started, his shyness, his awkwardness, his inarticulateness, fell away and he became much more productive in many other ways. Certainly he became much more articulate.

And I can cite another person in the same way. I trust he isn't in the audience, i.e. a fellow named Clogston. I met him when I was working on a problem with John Pierce's group and I didn't think he had much. I asked my friends who had been with him at school, "Was he like that in graduate school?" "Yes," they replied. Well I would have fired the fellow, but J. R. Pierce was smart and kept him on. Clogston finally did the Clogston cable. After that there was a steady stream of good ideas. One success brought him confidence and courage.

One of the characteristics of successful scientists is having courage. Once you get your courage up and believe that you can do important problems, then you can. If you think you can't, almost surely you are not going to. Courage is one of the things that Shannon had supremely. You have only to think of his major theorem. He wants to create a method of coding, but he doesn't know what to do so he makes a random code. Then he is stuck. And then he asks the impossible question, "What would the average random code do?" He then proves that the average code is arbitrarily good, and that therefore there must be at least one good code. Who but a man of infinite courage could have dared to think those thoughts? That is the characteristic of great scientists; they have courage. They will go forward under incredible circumstances; they think and continue to think.

Age is another factor which the physicists particularly worry about. They always are saying that you have got to do it when you are young or you will never do it. Einstein did things very early, and all the quantum mechanic fellows were disgustingly young when they did their best work. Most mathematicians, theoretical physicists, and astrophysicists do what we consider their best work when they are young. It is not that they don't do good work in their old age but what we value most is often what they did early. On the other hand, in music, politics and literature, often what we consider their best work was done late. I don't know how whatever field you are in fits this scale, but age has some effect.

But let me say why age seems to have the effect it does. In the first place if you do some good work you will find yourself on all kinds of committees and unable to do any more work. You may find yourself as I saw Brattain when he got a Nobel Prize. The day the prize was announced we all assembled in Arnold Auditorium; all three winners got up and made speeches. The third one, Brattain, practically with tears in his eyes, said, "I know about this Nobel-Prize effect and I am not going to let it affect me; I am going to remain good old Walter Brattain." Well I said to myself, "That is nice." But in a few weeks I saw it was affecting him. Now he could only work on great problems.

When you are famous it is hard to work on small problems. This is what did Shannon in. After information theory, what do you do for an encore? The great scientists often make this error. They fail to continue to plant the little acorns from which the mighty oak trees grow. They try to get the big thing right off. And that isn't the way things go. So that is another reason why you find that when you get early recognition it seems to sterilize you. In fact I will give you my favorite quotation of many years. The Institute for Advanced Study in Princeton, in my opinion, has ruined more good scientists than any institution has created, judged by what they did before they came and judged by what they did after. Not that they weren't good afterwards, but they were superb before they got there and were only good afterwards.

This brings up the subject, out of order perhaps, of working conditions. What most people think are the best working conditions, are not. Very clearly they are not because people are often most productive when working conditions are bad. One of the better times of the Cambridge Physical Laboratories was when they had practically shacks — they did some of the best physics ever.

I give you a story from my own private life. Early on it became evident to me that Bell Laboratories was not going to give me the conventional acre of programming people to program computing machines in absolute binary. It was clear they weren't going to. But that was the way everybody did it. I could go to the West Coast and get a job with the airplane companies without any trouble, but the exciting people were at Bell Labs and the fellows out there in the airplane companies were not. I thought for a long while about, "Did I want to go or not?" and I wondered how I could get the best of two possible worlds. I finally said to myself, "Hamming, you think the machines can do practically everything. Why can't you make them write programs?" What appeared at first to me as a defect forced me into automatic programming very early. What appears to be a fault, often, by a change of viewpoint, turns out to be one of the greatest assets you can have. But you are not likely to think that when you first look the thing and say, "Gee, I'm never going to get enough programmers, so how can I ever do any great programming?"

And there are many other stories of the same kind; Grace Hopper has similar ones. I think that if you look carefully you will see that often the great scientists, by turning the problem around a bit, changed a defect to an asset. For example, many scientists when they found they couldn't do a problem finally began to study why not. They then turned it around the other way and said, "But of course, this is what it is" and got an important result. So ideal working conditions are very strange. The ones you want aren't always the best ones for you.

Now for the matter of drive. You observe that most great scientists have tremendous drive. I worked for ten years with John Tukey at Bell Labs. He had tremendous drive. One day about three or four years after I joined, I discovered that John Tukey was slightly younger than I was. John was a genius and I clearly was not. Well I went storming into Bode's office and said, "How can anybody my age know as much as John Tukey does?" He leaned back in his chair, put his hands behind his head, grinned slightly, and said, "You would be surprised Hamming, how much you would know if you worked as hard as he did that many years." I simply slunk out of the office!

What Bode was saying was this: Knowledge and productivity are like compound interest. Given two people of approximately the same ability and one person who works ten percent more than the other, the latter will more than twice outproduce the former. The more you know, the more you learn; the more you learn, the more you can do; the more you can do, the more the opportunity — it is very much like compound interest. I don't want to give you a rate, but it is a very high rate. Given two people with exactly the same ability, the one person who manages day in and day out to get in one more hour of thinking will be tremendously more productive over a lifetime. I took Bode's remark to heart; I spent a good deal more of my time for some years trying to work a bit harder and I found, in fact, I could get more work done. I don't like to say it in front of my wife, but I did sort of neglect her sometimes; I needed to study. You have to neglect things if you intend to get what you want done. There's no question about this.

On this matter of drive Edison says, "Genius is 99% perspiration and 1% inspiration." He may have been exaggerating, but the idea is that solid work, steadily applied, gets you surprisingly far. The steady application of effort with a little bit more work, intelligently applied is what does it. That's the trouble; drive, misapplied, doesn't get you anywhere. I've often wondered why so many of my good friends at Bell Labs who worked as hard or harder than I did, didn't have so much to show for it. The misapplication of effort is a very serious matter. Just hard work is not enough - it must be applied sensibly.

There's another trait on the side which I want to talk about; that trait is ambiguity. It took me a while to discover its importance. Most people like to believe something is or is not true. Great scientists tolerate ambiguity very well. They believe the theory enough to go ahead; they doubt it enough to notice the errors and faults so they can step forward and create the new replacement theory. If you believe too much you'll never notice the flaws; if you doubt too much you won't get started. It requires a lovely balance. But most great scientists are well aware of why their theories are true and they are also well aware of some slight misfits which don't quite fit and they don't forget it. Darwin writes in his autobiography that he found it necessary to write down every piece of evidence which appeared to contradict his beliefs because otherwise they would disappear from his mind. When you find apparent flaws you've got to be sensitive and keep track of those things, and keep an eye out for how they can be explained or how the theory can be changed to fit them. Those are often the great contributions. Great contributions are rarely done by adding another decimal place. It comes down to an emotional commitment. Most great scientists are completely committed to their problem. Those who don't become committed seldom produce outstanding, first-class work.

Now again, emotional commitment is not enough. It is a necessary condition apparently. And I think I can tell you the reason why. Everybody who has studied creativity is driven finally to saying, "creativity comes out of your subconscious." Somehow, suddenly, there it is. It just appears. Well, we know very little about the subconscious; but one thing you are pretty well aware of is that your dreams also come out of your subconscious. And you're aware your dreams are, to a fair extent, a reworking of the experiences of the day. If you are deeply immersed and committed to a topic, day after day after day, your subconscious has nothing to do but work on your problem. And so you wake up one morning, or on some afternoon, and there's the answer. For those who don't get committed to their current problem, the subconscious goofs off on other things and doesn't produce the big result. So the way to manage yourself is that when you have a real important problem you don't let anything else get the center of your attention — you keep your thoughts on the problem. Keep your subconscious starved so it has to work on your problem, so you can sleep peacefully and get the answer in the morning, free.

Now Alan Chynoweth mentioned that I used to eat at the physics table. I had been eating with the mathematicians and I found out that I already knew a fair amount of mathematics; in fact, I wasn't learning much. The physics table was, as he said, an exciting place, but I think he exaggerated on how much I contributed. It was very interesting to listen to Shockley, Brattain, Bardeen, J. B. Johnson, Ken McKay and other people, and I was learning a lot. But unfortunately a Nobel Prize came, and a promotion came, and what was left was the dregs. Nobody wanted what was left. Well, there was no use eating with them!

Over on the other side of the dining hall was a chemistry table. I had worked with one of the fellows, Dave McCall; furthermore he was courting our secretary at the time. I went over and said, "Do you mind if I join you?" They can't say no, so I started eating with them for a while. And I started asking, "What are the important problems of your field?" And after a week or so, "What important problems are you working on?" And after some more time I came in one day and said, "If what you are doing is not important, and if you don't think it is going to lead to something important, why are you at Bell Labs working on it?" I wasn't welcomed after that; I had to find somebody else to eat with! That was in the spring.

In the fall, Dave McCall stopped me in the hall and said, "Hamming, that remark of yours got underneath my skin. I thought about it all summer, i.e. what were the important problems in my field. I haven't changed my research," he says, "but I think it was well worthwhile." And I said, "Thank you Dave," and went on. I noticed a couple of months later he was made the head of the department. I noticed the other day he was a Member of the National Academy of Engineering. I noticed he has succeeded. I have never heard the names of any of the other fellows at that table mentioned in science and scientific circles. They were unable to ask themselves, "What are the important problems in my field?"

If you do not work on an important problem, it's unlikely you'll do important work. It's perfectly obvious. Great scientists have thought through, in a careful way, a number of important problems in their field, and they keep an eye on wondering how to attack them. Let me warn you, "important problem" must be phrased carefully. The three outstanding problems in physics, in a certain sense, were never worked on while I was at Bell Labs. By important I mean guaranteed a Nobel Prize and any sum of money you want to mention. We didn't work on (1) time travel, (2) teleportation, and (3) antigravity. They are not important problems because we do not have an attack. It's not the consequence that makes a problem important, it is that you have a reasonable attack. That is what makes a problem important. When I say that most scientists don't work on important problems, I mean it in that sense. The average scientist, so far as I can make out, spends almost all his time working on problems which he believes will not be important and he also doesn't believe that they will lead to important problems.

I spoke earlier about planting acorns so that oaks will grow. You can't always know exactly where to be, but you can keep active in places where something might happen. And even if you believe that great science is a matter of luck, you can stand on a mountain top where lightning strikes; you don't have to hide in the valley where you're safe. But the average scientist does routine safe work almost all the time and so he (or she) doesn't produce much. It's that simple. If you want to do great work, you clearly must work on important problems, and you should have an idea.

Along those lines at some urging from John Tukey and others, I finally adopted what I called "Great Thoughts Time." When I went to lunch Friday noon, I would only discuss great thoughts after that. By great thoughts I mean ones like: "What will be the role of computers in all of AT&T?", "How will computers change science?" For example, I came up with the observation at that time that nine out of ten experiments were done in the lab and one in ten on the computer. I made a remark to the vice presidents one time, that it would be reversed, i.e. nine out of ten experiments would be done on the computer and one in ten in the lab. They knew I was a crazy mathematician and had no sense of reality. I knew they were wrong and they've been proved wrong while I have been proved right. They built laboratories when they didn't need them. I saw that computers were transforming science because I spent a lot of time asking "What will be the impact of computers on science and how can I change it?" I asked myself, "How is it going to change Bell Labs?" I remarked one time, in the same address, that more than one-half of the people at Bell Labs will be interacting closely with computing machines before I leave. Well, you all have terminals now. I thought hard about where was my field going, where were the opportunities, and what were the important things to do. Let me go there so there is a chance I can do important things.

Most great scientists know many important problems. They have something between 10 and 20 important problems for which they are looking for an attack. And when they see a new idea come up, one hears them say "Well that bears on this problem." They drop all the other things and get after it. Now I can tell you a horror story that was told to me but I can't vouch for the truth of it. I was sitting in an airport talking to a friend of mine from Los Alamos about how it was lucky that the fission experiment occurred over in Europe when it did because that got us working on the atomic bomb here in the US. He said "No; at Berkeley we had gathered a bunch of data; we didn't get around to reducing it because we were building some more equipment, but if we had reduced that data we would have found fission." They had it in their hands and they didn't pursue it. They came in second!

The great scientists, when an opportunity opens up, get after it and they pursue it. They drop all other things. They get rid of other things and they get after an idea because they had already thought the thing through. Their minds are prepared; they see the opportunity and they go after it. Now of course lots of times it doesn't work out, but you don't have to hit many of them to do some great science. It's kind of easy. One of the chief tricks is to live a long time!

Another trait, it took me a while to notice. I noticed the following facts about people who work with the door open or the door closed. I notice that if you have the door to your office closed, you get more work done today and tomorrow, and you are more productive than most. But 10 years later somehow you don't know quite know what problems are worth working on; all the hard work you do is sort of tangential in importance. He who works with the door open gets all kinds of interruptions, but he also occasionally gets clues as to what the world is and what might be important. Now I cannot prove the cause and effect sequence because you might say, "The closed door is symbolic of a closed mind." I don't know. But I can say there is a pretty good correlation between those who work with the doors open and those who ultimately do important things, although people who work with doors closed often work harder. Somehow they seem to work on slightly the wrong thing — not much, but enough that they miss fame.

I want to talk on another topic. It is based on the song which I think many of you know, "It ain't what you do, it's the way that you do it." I'll start with an example of my own. I was conned into doing on a digital computer, in the absolute binary days, a problem which the best analog computers couldn't do. And I was getting an answer. When I thought carefully and said to myself, "You know, Hamming, you're going to have to file a report on this military job; after you spend a lot of money you're going to have to account for it and every analog installation is going to want the report to see if they can't find flaws in it." I was doing the required integration by a rather crummy method, to say the least, but I was getting the answer. And I realized that in truth the problem was not just to get the answer; it was to demonstrate for the first time, and beyond question, that I could beat the analog computer on its own ground with a digital machine. I reworked the method of solution, created a theory which was nice and elegant, and changed the way we computed the answer; the results were no different. The published report had an elegant method which was later known for years as "Hamming's Method of Integrating Differential Equations." It is somewhat obsolete now, but for a while it was a very good method. By changing the problem slightly, I did important work rather than trivial work.

In the same way, when using the machine up in the attic in the early days, I was solving one problem after another after another; a fair number were successful and there were a few failures. I went home one Friday after finishing a problem, and curiously enough I wasn't happy; I was depressed. I could see life being a long sequence of one problem after another after another. After quite a while of thinking I decided, "No, I should be in the mass production of a variable product. I should be concerned with all of next year's problems, not just the one in front of my face." By changing the question I still got the same kind of results or better, but I changed things and did important work. I attacked the major problem — How do I conquer machines and do all of next year's problems when I don't know what they are going to be? How do I prepare for it? How do I do this one so I'll be on top of it? How do I obey Newton's rule? He said, "If I have seen further than others, it is because I've stood on the shoulders of giants." These days we stand on each other's feet!

You should do your job in such a fashion that others can build on top of it, so they will indeed say, "Yes, I've stood on so and so's shoulders and I saw further." The essence of science is cumulative. By changing a problem slightly you can often do great work rather than merely good work. Instead of attacking isolated problems, I made the resolution that I would never again solve an isolated problem except as characteristic of a class.

Now if you are much of a mathematician you know that the effort to generalize often means that the solution is simple. Often by stopping and saying, "This is the problem he wants but this is characteristic of so and so. Yes, I can attack the whole class with a far superior method than the particular one because I was earlier embedded in needless detail." The business of abstraction frequently makes things simple. Furthermore, I filed away the methods and prepared for the future problems.

To end this part, I'll remind you, "It is a poor workman who blames his tools — the good man gets on with the job, given what he's got, and gets the best answer he can." And I suggest that by altering the problem, by looking at the thing differently, you can make a great deal of difference in your final productivity because you can either do it in such a fashion that people can indeed build on what you've done, or you can do it in such a fashion that the next person has to essentially duplicate again what you've done. It isn't just a matter of the job, it's the way you write the report, the way you write the paper, the whole attitude. It's just as easy to do a broad, general job as one very special case. And it's much more satisfying and rewarding!

I have now come down to a topic which is very distasteful; it is not sufficient to do a job, you have to sell it. "Selling" to a scientist is an awkward thing to do. It's very ugly; you shouldn't have to do it. The world is supposed to be waiting, and when you do something great, they should rush out and welcome it. But the fact is everyone is busy with their own work. You must present it so well that they will set aside what they are doing, look at what you've done, read it, and come back and say, "Yes, that was good." I suggest that when you open a journal, as you turn the pages, you ask why you read some articles and not others. You had better write your report so when it is published in the Physical Review, or wherever else you want it, as the readers are turning the pages they won't just turn your pages but they will stop and read yours. If they don't stop and read it, you won't get credit.

There are three things you have to do in selling. You have to learn to write clearly and well so that people will read it, you must learn to give reasonably formal talks, and you also must learn to give informal talks. We had a lot of so-called `back room scientists.' In a conference, they would keep quiet. Three weeks later after a decision was made they filed a report saying why you should do so and so. Well, it was too late. They would not stand up right in the middle of a hot conference, in the middle of activity, and say, "We should do this for these reasons." You need to master that form of communication as well as prepared speeches.

When I first started, I got practically physically ill while giving a speech, and I was very, very nervous. I realized I either had to learn to give speeches smoothly or I would essentially partially cripple my whole career. The first time IBM asked me to give a speech in New York one evening, I decided I was going to give a really good speech, a speech that was wanted, not a technical one but a broad one, and at the end if they liked it, I'd quietly say, "Any time you want one I'll come in and give you one." As a result, I got a great deal of practice giving speeches to a limited audience and I got over being afraid. Furthermore, I could also then study what methods were effective and what were ineffective.

While going to meetings I had already been studying why some papers are remembered and most are not. The technical person wants to give a highly limited technical talk. Most of the time the audience wants a broad general talk and wants much more survey and background than the speaker is willing to give. As a result, many talks are ineffective. The speaker names a topic and suddenly plunges into the details he's solved. Few people in the audience may follow. You should paint a general picture to say why it's important, and then slowly give a sketch of what was done. Then a larger number of people will say, "Yes, Joe has done that," or "Mary has done that; I really see where it is; yes, Mary really gave a good talk; I understand what Mary has done." The tendency is to give a highly restricted, safe talk; this is usually ineffective. Furthermore, many talks are filled with far too much information. So I say this idea of selling is obvious.

Let me summarize. You've got to work on important problems. I deny that it is all luck, but I admit there is a fair element of luck. I subscribe to Pasteur's "Luck favors the prepared mind." I favor heavily what I did. Friday afternoons for years — great thoughts only — means that I committed 10% of my time trying to understand the bigger problems in the field, i.e. what was and what was not important. I found in the early days I had believed `this' and yet had spent all week marching in `that' direction. It was kind of foolish. If I really believe the action is over there, why do I march in this direction? I either had to change my goal or change what I did. So I changed something I did and I marched in the direction I thought was important. It's that easy.

Now you might tell me you haven't got control over what you have to work on. Well, when you first begin, you may not. But once you're moderately successful, there are more people asking for results than you can deliver and you have some power of choice, but not completely. I'll tell you a story about that, and it bears on the subject of educating your boss. I had a boss named Schelkunoff; he was, and still is, a very good friend of mine. Some military person came to me and demanded some answers by Friday. Well, I had already dedicated my computing resources to reducing data on the fly for a group of scientists; I was knee deep in short, small, important problems. This military person wanted me to solve his problem by the end of the day on Friday. I said, "No, I'll give it to you Monday. I can work on it over the weekend. I'm not going to do it now." He goes down to my boss, Schelkunoff, and Schelkunoff says, "You must run this for him; he's got to have it by Friday." I tell him, "Why do I?" He says, "You have to." I said, "Fine, Sergei, but you're sitting in your office Friday afternoon catching the late bus home to watch as this fellow walks out that door." I gave the military person the answers late Friday afternoon. I then went to Schelkunoff's office and sat down; as the man goes out I say, "You see Schelkunoff, this fellow has nothing under his arm; but I gave him the answers." On Monday morning Schelkunoff called him up and said, "Did you come in to work over the weekend?" I could hear, as it were, a pause as the fellow ran through his mind of what was going to happen; but he knew he would have had to sign in, and he'd better not say he had when he hadn't, so he said he hadn't. Ever after that Schelkunoff said, "You set your deadlines; you can change them."

One lesson was sufficient to educate my boss as to why I didn't want to do big jobs that displaced exploratory research and why I was justified in not doing crash jobs which absorb all the research computing facilities. I wanted instead to use the facilities to compute a large number of small problems. Again, in the early days, I was limited in computing capacity and it was clear, in my area, that a "mathematician had no use for machines." But I needed more machine capacity. Every time I had to tell some scientist in some other area, "No I can't; I haven't the machine capacity," he complained. I said "Go tell your Vice President that Hamming needs more computing capacity." After a while I could see what was happening up there at the top; many people said to my Vice President, "Your man needs more computing capacity." I got it!

I also did a second thing. When I loaned what little programming power we had to help in the early days of computing, I said, "We are not getting the recognition for our programmers that they deserve. When you publish a paper you will thank that programmer or you aren't getting any more help from me. That programmer is going to be thanked by name; she's worked hard." I waited a couple of years. I then went through a year of BSTJ articles and counted what fraction thanked some programmer. I took it into the boss and said, "That's the central role computing is playing in Bell Labs; if the BSTJ is important, that's how important computing is." He had to give in. You can educate your bosses. It's a hard job. In this talk I'm only viewing from the bottom up; I'm not viewing from the top down. But I am telling you how you can get what you want in spite of top management. You have to sell your ideas there also.

Well I now come down to the topic, "Is the effort to be a great scientist worth it?" To answer this, you must ask people. When you get beyond their modesty, most people will say, "Yes, doing really first-class work, and knowing it, is as good as wine, women and song put together," or if it's a woman she says, "It is as good as wine, men and song put together." And if you look at the bosses, they tend to come back or ask for reports, trying to participate in those moments of discovery. They're always in the way. So evidently those who have done it, want to do it again. But it is a limited survey. I have never dared to go out and ask those who didn't do great work how they felt about the matter. It's a biased sample, but I still think it is worth the struggle. I think it is very definitely worth the struggle to try and do first-class work because the truth is, the value is in the struggle more than it is in the result. The struggle to make something of yourself seems to be worthwhile in itself. The success and fame are sort of dividends, in my opinion.

I've told you how to do it. It is so easy, so why do so many people, with all their talents, fail? For example, my opinion, to this day, is that there are in the mathematics department at Bell Labs quite a few people far more able and far better endowed than I, but they didn't produce as much. Some of them did produce more than I did; Shannon produced more than I did, and some others produced a lot, but I was highly productive against a lot of other fellows who were better equipped. Why is it so? What happened to them? Why do so many of the people who have great promise, fail?

Well, one of the reasons is drive and commitment. The people who do great work with less ability but who are committed to it, get more done that those who have great skill and dabble in it, who work during the day and go home and do other things and come back and work the next day. They don't have the deep commitment that is apparently necessary for really first-class work. They turn out lots of good work, but we were talking, remember, about first-class work. There is a difference. Good people, very talented people, almost always turn out good work. We're talking about the outstanding work, the type of work that gets the Nobel Prize and gets recognition.

The second thing is, I think, the problem of personality defects. Now I'll cite a fellow whom I met out in Irvine. He had been the head of a computing center and he was temporarily on assignment as a special assistant to the president of the university. It was obvious he had a job with a great future. He took me into his office one time and showed me his method of getting letters done and how he took care of his correspondence. He pointed out how inefficient the secretary was. He kept all his letters stacked around there; he knew where everything was. And he would, on his word processor, get the letter out. He was bragging how marvelous it was and how he could get so much more work done without the secretary's interference. Well, behind his back, I talked to the secretary. The secretary said, "Of course I can't help him; I don't get his mail. He won't give me the stuff to log in; I don't know where he puts it on the floor. Of course I can't help him." So I went to him and said, "Look, if you adopt the present method and do what you can do single-handedly, you can go just that far and no farther than you can do single-handedly. If you will learn to work with the system, you can go as far as the system will support you." And, he never went any further. He had his personality defect of wanting total control and was not willing to recognize that you need the support of the system.

You find this happening again and again; good scientists will fight the system rather than learn to work with the system and take advantage of all the system has to offer. It has a lot, if you learn how to use it. It takes patience, but you can learn how to use the system pretty well, and you can learn how to get around it. After all, if you want a decision `No', you just go to your boss and get a `No' easy. If you want to do something, don't ask, do it. Present him with an accomplished fact. Don't give him a chance to tell you `No'. But if you want a `No', it's easy to get a `No'.

Another personality defect is ego assertion and I'll speak in this case of my own experience. I came from Los Alamos and in the early days I was using a machine in New York at 590 Madison Avenue where we merely rented time. I was still dressing in western clothes, big slash pockets, a bolo and all those things. I vaguely noticed that I was not getting as good service as other people. So I set out to measure. You came in and you waited for your turn; I felt I was not getting a fair deal. I said to myself, "Why? No Vice President at IBM said, `Give Hamming a bad time'. It is the secretaries at the bottom who are doing this. When a slot appears, they'll rush to find someone to slip in, but they go out and find somebody else. Now, why? I haven't mistreated them." Answer: I wasn't dressing the way they felt somebody in that situation should. It came down to just that — I wasn't dressing properly. I had to make the decision — was I going to assert my ego and dress the way I wanted to and have it steadily drain my effort from my professional life, or was I going to appear to conform better? I decided I would make an effort to appear to conform properly. The moment I did, I got much better service. And now, as an old colorful character, I get better service than other people.

You should dress according to the expectations of the audience spoken to. If I am going to give an address at the MIT computer center, I dress with a bolo and an old corduroy jacket or something else. I know enough not to let my clothes, my appearance, my manners get in the way of what I care about. An enormous number of scientists feel they must assert their ego and do their thing their way. They have got to be able to do this, that, or the other thing, and they pay a steady price.

John Tukey almost always dressed very casually. He would go into an important office and it would take a long time before the other fellow realized that this is a first-class man and he had better listen. For a long time John has had to overcome this kind of hostility. It's wasted effort! I didn't say you should conform; I said "The appearance of conforming gets you a long way." If you chose to assert your ego in any number of ways, "I am going to do it my way," you pay a small steady price throughout the whole of your professional career. And this, over a whole lifetime, adds up to an enormous amount of needless trouble.

By taking the trouble to tell jokes to the secretaries and being a little friendly, I got superb secretarial help. For instance, one time for some idiot reason all the reproducing services at Murray Hill were tied up. Don't ask me how, but they were. I wanted something done. My secretary called up somebody at Holmdel, hopped the company car, made the hour-long trip down and got it reproduced, and then came back. It was a payoff for the times I had made an effort to cheer her up, tell her jokes and be friendly; it was that little extra work that later paid off for me. By realizing you have to use the system and studying how to get the system to do your work, you learn how to adapt the system to your desires. Or you can fight it steadily, as a small undeclared war, for the whole of your life.

And I think John Tukey paid a terrible price needlessly. He was a genius anyhow, but I think it would have been far better, and far simpler, had he been willing to conform a little bit instead of ego asserting. He is going to dress the way he wants all of the time. It applies not only to dress but to a thousand other things; people will continue to fight the system. Not that you shouldn't occasionally!

When they moved the library from the middle of Murray Hill to the far end, a friend of mine put in a request for a bicycle. Well, the organization was not dumb. They waited awhile and sent back a map of the grounds saying, "Will you please indicate on this map what paths you are going to take so we can get an insurance policy covering you." A few more weeks went by. They then asked, "Where are you going to store the bicycle and how will it be locked so we can do so and so." He finally realized that of course he was going to be red-taped to death so he gave in. He rose to be the President of Bell Laboratories.

Barney Oliver was a good man. He wrote a letter one time to the IEEE. At that time the official shelf space at Bell Labs was so much and the height of the IEEE Proceedings at that time was larger; and since you couldn't change the size of the official shelf space he wrote this letter to the IEEE Publication person saying, since so many IEEE members were at Bell Labs and since the official space was so high the journal size should be changed. He sent it for his boss's signature. Back came a carbon with his signature, but he still doesn't know whether the original was sent or not. I am not saying you shouldn't make gestures of reform. I am saying that my study of able people is that they don't get themselves committed to that kind of warfare. They play it a little bit and drop it and get on with their work.

Many a second-rate fellow gets caught up in some little twitting of the system, and carries it through to warfare. He expends his energy in a foolish project. Now you are going to tell me that somebody has to change the system. I agree; somebody's has to. Which do you want to be? The person who changes the system or the person who does first-class science? Which person is it that you want to be? Be clear, when you fight the system and struggle with it, what you are doing, how far to go out of amusement, and how much to waste your effort fighting the system. My advice is to let somebody else do it and you get on with becoming a first-class scientist. Very few of you have the ability to both reform the system and become a first-class scientist.

On the other hand, we can't always give in. There are times when a certain amount of rebellion is sensible. I have observed almost all scientists enjoy a certain amount of twitting the system for the sheer love of it. What it comes down to basically is that you cannot be original in one area without having originality in others. Originality is being different. You can't be an original scientist without having some other original characteristics. But many a scientist has let his quirks in other places make him pay a far higher price than is necessary for the ego satisfaction he or she gets. I'm not against all ego assertion; I'm against some.

Another fault is anger. Often a scientist becomes angry, and this is no way to handle things. Amusement, yes, anger, no. Anger is misdirected. You should follow and cooperate rather than struggle against the system all the time.

Another thing you should look for is the positive side of things instead of the negative. I have already given you several examples, and there are many, many more; how, given the situation, by changing the way I looked at it, I converted what was apparently a defect to an asset. I'll give you another example. I am an egotistical person; there is no doubt about it. I knew that most people who took a sabbatical to write a book, didn't finish it on time. So before I left, I told all my friends that when I come back, that book was going to be done! Yes, I would have it done — I'd have been ashamed to come back without it! I used my ego to make myself behave the way I wanted to. I bragged about something so I'd have to perform. I found out many times, like a cornered rat in a real trap, I was surprisingly capable. I have found that it paid to say, ``Oh yes, I'll get the answer for you Tuesday,'' not having any idea how to do it. By Sunday night I was really hard thinking on how I was going to deliver by Tuesday. I often put my pride on the line and sometimes I failed, but as I said, like a cornered rat I'm surprised how often I did a good job. I think you need to learn to use yourself. I think you need to know how to convert a situation from one view to another which would increase the chance of success.

Now self-delusion in humans is very, very common. There are innumerable ways of you changing a thing and kidding yourself and making it look some other way. When you ask, "Why didn't you do such and such," the person has a thousand alibis. If you look at the history of science, usually these days there are ten people right there ready, and we pay off for the person who is there first. The other nine fellows say, "Well, I had the idea but I didn't do it and so on and so on." There are so many alibis. Why weren't you first? Why didn't you do it right? Don't try an alibi. Don't try and kid yourself. You can tell other people all the alibis you want. I don't mind. But to yourself try to be honest.

If you really want to be a first-class scientist you need to know yourself, your weaknesses, your strengths, and your bad faults, like my egotism. How can you convert a fault to an asset? How can you convert a situation where you haven't got enough manpower to move into a direction when that's exactly what you need to do? I say again that I have seen, as I studied the history, the successful scientist changed the viewpoint and what was a defect became an asset.

In summary, I claim that some of the reasons why so many people who have greatness within their grasp don't succeed are: they don't work on important problems, they don't become emotionally involved, they don't try and change what is difficult to some other situation which is easily done but is still important, and they keep giving themselves alibis why they don't. They keep saying that it is a matter of luck. I've told you how easy it is; furthermore I've told you how to reform. Therefore, go forth and become great scientists!



Questions and Answers

A. G. Chynoweth: Well that was 50 minutes of concentrated wisdom and observations accumulated over a fantastic career; I lost track of all the observations that were striking home. Some of them are very very timely. One was the plea for more computer capacity; I was hearing nothing but that this morning from several people, over and over again. So that was right on the mark today even though here we are 20 – 30 years after when you were making similar remarks, Dick. I can think of all sorts of lessons that all of us can draw from your talk. And for one, as I walk around the halls in the future I hope I won't see as many closed doors in Bellcore. That was one observation I thought was very intriguing.

Thank you very, very much indeed Dick; that was a wonderful recollection. I'll now open it up for questions. I'm sure there are many people who would like to take up on some of the points that Dick was making.

Hamming: First let me respond to Alan Chynoweth about computing. I had computing in research and for 10 years I kept telling my management, ``Get that !&@#% machine out of research. We are being forced to run problems all the time. We can't do research because were too busy operating and running the computing machines.'' Finally the message got through. They were going to move computing out of research to someplace else. I was persona non grata to say the least and I was surprised that people didn't kick my shins because everybody was having their toy taken away from them. I went in to Ed David's office and said, ``Look Ed, you've got to give your researchers a machine. If you give them a great big machine, we'll be back in the same trouble we were before, so busy keeping it going we can't think. Give them the smallest machine you can because they are very able people. They will learn how to do things on a small machine instead of mass computing.'' As far as I'm concerned, that's how UNIX arose. We gave them a moderately small machine and they decided to make it do great things. They had to come up with a system to do it on. It is called UNIX!

A. G. Chynoweth: I just have to pick up on that one. In our present environment, Dick, while we wrestle with some of the red tape attributed to, or required by, the regulators, there is one quote that one exasperated AVP came up with and I've used it over and over again. He growled that, "UNIX was never a deliverable!"

Question: What about personal stress? Does that seem to make a difference?

Hamming: Yes, it does. If you don't get emotionally involved, it doesn't. I had incipient ulcers most of the years that I was at Bell Labs. I have since gone off to the Naval Postgraduate School and laid back somewhat, and now my health is much better. But if you want to be a great scientist you're going to have to put up with stress. You can lead a nice life; you can be a nice guy or you can be a great scientist. But nice guys end last, is what Leo Durocher said. If you want to lead a nice happy life with a lot of recreation and everything else, you'll lead a nice life.

Question: The remarks about having courage, no one could argue with; but those of us who have gray hairs or who are well established don't have to worry too much. But what I sense among the young people these days is a real concern over the risk taking in a highly competitive environment. Do you have any words of wisdom on this?

Hamming: I'll quote Ed David more. Ed David was concerned about the general loss of nerve in our society. It does seem to me that we've gone through various periods. Coming out of the war, coming out of Los Alamos where we built the bomb, coming out of building the radars and so on, there came into the mathematics department, and the research area, a group of people with a lot of guts. They've just seen things done; they've just won a war which was fantastic. We had reasons for having courage and therefore we did a great deal. I can't arrange that situation to do it again. I cannot blame the present generation for not having it, but I agree with what you say; I just cannot attach blame to it. It doesn't seem to me they have the desire for greatness; they lack the courage to do it. But we had, because we were in a favorable circumstance to have it; we just came through a tremendously successful war. In the war we were looking very, very bad for a long while; it was a very desperate struggle as you well know. And our success, I think, gave us courage and self confidence; that's why you see, beginning in the late forties through the fifties, a tremendous productivity at the labs which was stimulated from the earlier times. Because many of us were earlier forced to learn other things — we were forced to learn the things we didn't want to learn, we were forced to have an open door — and then we could exploit those things we learned. It is true, and I can't do anything about it; I cannot blame the present generation either. It's just a fact.

Question: Is there something management could or should do?

Hamming: Management can do very little. If you want to talk about managing research, that's a totally different talk. I'd take another hour doing that. This talk is about how the individual gets very successful research done in spite of anything the management does or in spite of any other opposition. And how do you do it? Just as I observe people doing it. It's just that simple and that hard!

Question: Is brainstorming a daily process?

Hamming: Once that was a very popular thing, but it seems not to have paid off. For myself I find it desirable to talk to other people; but a session of brainstorming is seldom worthwhile. I do go in to strictly talk to somebody and say, "Look, I think there has to be something here. Here's what I think I see ..." and then begin talking back and forth. But you want to pick capable people. To use another analogy, you know the idea called the `critical mass.' If you have enough stuff you have critical mass. There is also the idea I used to call `sound absorbers'. When you get too many sound absorbers, you give out an idea and they merely say, "Yes, yes, yes." What you want to do is get that critical mass in action; "Yes, that reminds me of so and so," or, "Have you thought about that or this?" When you talk to other people, you want to get rid of those sound absorbers who are nice people but merely say, "Oh yes," and to find those who will stimulate you right back.

For example, you couldn't talk to John Pierce without being stimulated very quickly. There were a group of other people I used to talk with. For example there was Ed Gilbert; I used to go down to his office regularly and ask him questions and listen and come back stimulated. I picked my people carefully with whom I did or whom I didn't brainstorm because the sound absorbers are a curse. They are just nice guys; they fill the whole space and they contribute nothing except they absorb ideas and the new ideas just die away instead of echoing on. Yes, I find it necessary to talk to people. I think people with closed doors fail to do this so they fail to get their ideas sharpened, such as "Did you ever notice something over here?" I never knew anything about it — I can go over and look. Somebody points the way. On my visit here, I have already found several books that I must read when I get home. I talk to people and ask questions when I think they can answer me and give me clues that I do not know about. I go out and look!

Question: What kind of tradeoffs did you make in allocating your time for reading and writing and actually doing research?

Hamming: I believed, in my early days, that you should spend at least as much time in the polish and presentation as you did in the original research. Now at least 50% of the time must go for the presentation. It's a big, big number.

Question: How much effort should go into library work?

Hamming: It depends upon the field. I will say this about it. There was a fellow at Bell Labs, a very, very, smart guy. He was always in the library; he read everything. If you wanted references, you went to him and he gave you all kinds of references. But in the middle of forming these theories, I formed a proposition: there would be no effect named after him in the long run. He is now retired from Bell Labs and is an Adjunct Professor. He was very valuable; I'm not questioning that. He wrote some very good Physical Review articles; but there's no effect named after him because he read too much. If you read all the time what other people have done you will think the way they thought. If you want to think new thoughts that are different, then do what a lot of creative people do — get the problem reasonably clear and then refuse to look at any answers until you've thought the problem through carefully how you would do it, how you could slightly change the problem to be the correct one. So yes, you need to keep up. You need to keep up more to find out what the problems are than to read to find the solutions. The reading is necessary to know what is going on and what is possible. But reading to get the solutions does not seem to be the way to do great research. So I'll give you two answers. You read; but it is not the amount, it is the way you read that counts.

Question: How do you get your name attached to things?

Hamming: By doing great work. I'll tell you the hamming window one. I had given Tukey a hard time, quite a few times, and I got a phone call from him from Princeton to me at Murray Hill. I knew that he was writing up power spectra and he asked me if I would mind if he called a certain window a "hamming window." And I said to him, "Come on, John; you know perfectly well I did only a small part of the work but you also did a lot." He said, "Yes, Hamming, but you contributed a lot of small things; you're entitled to some credit." So he called it the hamming window. Now, let me go on. I had twitted John frequently about true greatness. I said true greatness is when your name is like ampere, watt, and fourier — when it's spelled with a lower case letter. That's how the hamming window came about.

Question: Dick, would you care to comment on the relative effectiveness between giving talks, writing papers, and writing books?

Hamming: In the short-haul, papers are very important if you want to stimulate someone tomorrow. If you want to get recognition long-haul, it seems to me writing books is more contribution because most of us need orientation. In this day of practically infinite knowledge, we need orientation to find our way. Let me tell you what infinite knowledge is. Since from the time of Newton to now, we have come close to doubling knowledge every 17 years, more or less. And we cope with that, essentially, by specialization. In the next 340 years at that rate, there will be 20 doublings, i.e. a million, and there will be a million fields of specialty for every one field now. It isn't going to happen. The present growth of knowledge will choke itself off until we get different tools. I believe that books which try to digest, coordinate, get rid of the duplication, get rid of the less fruitful methods and present the underlying ideas clearly of what we know now, will be the things the future generations will value. Public talks are necessary; private talks are necessary; written papers are necessary. But I am inclined to believe that, in the long-haul, books which leave out what's not essential are more important than books which tell you everything because you don't want to know everything. I don't want to know that much about penguins is the usual reply. You just want to know the essence.

Question: You mentioned the problem of the Nobel Prize and the subsequent notoriety of what was done to some of the careers. Isn't that kind of a much more broad problem of fame? What can one do?

Hamming: Some things you could do are the following. Somewhere around every seven years make a significant, if not complete, shift in your field. Thus, I shifted from numerical analysis, to hardware, to software, and so on, periodically, because you tend to use up your ideas. When you go to a new field, you have to start over as a baby. You are no longer the big mukity muk and you can start back there and you can start planting those acorns which will become the giant oaks. Shannon, I believe, ruined himself. In fact when he left Bell Labs, I said, "That's the end of Shannon's scientific career." I received a lot of flak from my friends who said that Shannon was just as smart as ever. I said, "Yes, he'll be just as smart, but that's the end of his scientific career," and I truly believe it was.

You have to change. You get tired after a while; you use up your originality in one field. You need to get something nearby. I'm not saying that you shift from music to theoretical physics to English literature; I mean within your field you should shift areas so that you don't go stale. You couldn't get away with forcing a change every seven years, but if you could, I would require a condition for doing research, being that you will change your field of research every seven years with a reasonable definition of what it means, or at the end of 10 years, management has the right to compel you to change. I would insist on a change because I'm serious. What happens to the old fellows is that they get a technique going; they keep on using it. They were marching in that direction which was right then, but the world changes. There's the new direction; but the old fellows are still marching in their former direction.

You need to get into a new field to get new viewpoints, and before you use up all the old ones. You can do something about this, but it takes effort and energy. It takes courage to say, ``Yes, I will give up my great reputation.'' For example, when error correcting codes were well launched, having these theories, I said, "Hamming, you are going to quit reading papers in the field; you are going to ignore it completely; you are going to try and do something else other than coast on that." I deliberately refused to go on in that field. I wouldn't even read papers to try to force myself to have a chance to do something else. I managed myself, which is what I'm preaching in this whole talk. Knowing many of my own faults, I manage myself. I have a lot of faults, so I've got a lot of problems, i.e. a lot of possibilities of management.

Question: Would you compare research and management?

Hamming: If you want to be a great researcher, you won't make it being president of the company. If you want to be president of the company, that's another thing. I'm not against being president of the company. I just don't want to be. I think Ian Ross does a good job as President of Bell Labs. I'm not against it; but you have to be clear on what you want. Furthermore, when you're young, you may have picked wanting to be a great scientist, but as you live longer, you may change your mind. For instance, I went to my boss, Bode, one day and said, "Why did you ever become department head? Why didn't you just be a good scientist?" He said, "Hamming, I had a vision of what mathematics should be in Bell Laboratories. And I saw if that vision was going to be realized, I had to make it happen; I had to be department head." When your vision of what you want to do is what you can do single-handedly, then you should pursue it. The day your vision, what you think needs to be done, is bigger than what you can do single-handedly, then you have to move toward management. And the bigger the vision is, the farther in management you have to go. If you have a vision of what the whole laboratory should be, or the whole Bell System, you have to get there to make it happen. You can't make it happen from the bottom very easily. It depends upon what goals and what desires you have. And as they change in life, you have to be prepared to change. I chose to avoid management because I preferred to do what I could do single-handedly. But that's the choice that I made, and it is biased. Each person is entitled to their choice. Keep an open mind. But when you do choose a path, for heaven's sake be aware of what you have done and the choice you have made. Don't try to do both sides.

Question: How important is one's own expectation or how important is it to be in a group or surrounded by people who expect great work from you?

Hamming: At Bell Labs everyone expected good work from me — it was a big help. Everybody expects you to do a good job, so you do, if you've got pride. I think it's very valuable to have first-class people around. I sought out the best people. The moment that physics table lost the best people, I left. The moment I saw that the same was true of the chemistry table, I left. I tried to go with people who had great ability so I could learn from them and who would expect great results out of me. By deliberately managing myself, I think I did much better than laissez faire.

Question: You, at the outset of your talk, minimized or played down luck; but you seemed also to gloss over the circumstances that got you to Los Alamos, that got you to Chicago, that got you to Bell Laboratories.

Hamming: There was some luck. On the other hand I don't know the alternate branches. Until you can say that the other branches would not have been equally or more successful, I can't say. Is it luck the particular thing you do? For example, when I met Feynman at Los Alamos, I knew he was going to get a Nobel Prize. I didn't know what for. But I knew darn well he was going to do great work. No matter what directions came up in the future, this man would do great work. And sure enough, he did do great work. It isn't that you only do a little great work at this circumstance and that was luck, there are many opportunities sooner or later. There are a whole pail full of opportunities, of which, if you're in this situation, you seize one and you're great over there instead of over here. There is an element of luck, yes and no. Luck favors a prepared mind; luck favors a prepared person. It is not guaranteed; I don't guarantee success as being absolutely certain. I'd say luck changes the odds, but there is some definite control on the part of the individual.

Go forth, then, and do great work!

"""



pg = """If you collected lists of techniques for doing great work in a lot of different fields, what would the intersection look like? I decided to find out by making it.

Partly my goal was to create a guide that could be used by someone working in any field. But I was also curious about the shape of the intersection. And one thing this exercise shows is that it does have a definite shape; it's not just a point labelled "work hard."

The following recipe assumes you're very ambitious.





The first step is to decide what to work on. The work you choose needs to have three qualities: it has to be something you have a natural aptitude for, that you have a deep interest in, and that offers scope to do great work.

In practice you don't have to worry much about the third criterion. Ambitious people are if anything already too conservative about it. So all you need to do is find something you have an aptitude for and great interest in. [1]

That sounds straightforward, but it's often quite difficult. When you're young you don't know what you're good at or what different kinds of work are like. Some kinds of work you end up doing may not even exist yet. So while some people know what they want to do at 14, most have to figure it out.

The way to figure out what to work on is by working. If you're not sure what to work on, guess. But pick something and get going. You'll probably guess wrong some of the time, but that's fine. It's good to know about multiple things; some of the biggest discoveries come from noticing connections between different fields.

Develop a habit of working on your own projects. Don't let "work" mean something other people tell you to do. If you do manage to do great work one day, it will probably be on a project of your own. It may be within some bigger project, but you'll be driving your part of it.

What should your projects be? Whatever seems to you excitingly ambitious. As you grow older and your taste in projects evolves, exciting and important will converge. At 7 it may seem excitingly ambitious to build huge things out of Lego, then at 14 to teach yourself calculus, till at 21 you're starting to explore unanswered questions in physics. But always preserve excitingness.

There's a kind of excited curiosity that's both the engine and the rudder of great work. It will not only drive you, but if you let it have its way, will also show you what to work on.

What are you excessively curious about — curious to a degree that would bore most other people? That's what you're looking for.

Once you've found something you're excessively interested in, the next step is to learn enough about it to get you to one of the frontiers of knowledge. Knowledge expands fractally, and from a distance its edges look smooth, but once you learn enough to get close to one, they turn out to be full of gaps.

The next step is to notice them. This takes some skill, because your brain wants to ignore such gaps in order to make a simpler model of the world. Many discoveries have come from asking questions about things that everyone else took for granted. [2]

If the answers seem strange, so much the better. Great work often has a tincture of strangeness. You see this from painting to math. It would be affected to try to manufacture it, but if it appears, embrace it.

Boldly chase outlier ideas, even if other people aren't interested in them — in fact, especially if they aren't. If you're excited about some possibility that everyone else ignores, and you have enough expertise to say precisely what they're all overlooking, that's as good a bet as you'll find. [3]

Four steps: choose a field, learn enough to get to the frontier, notice gaps, explore promising ones. This is how practically everyone who's done great work has done it, from painters to physicists.

Steps two and four will require hard work. It may not be possible to prove that you have to work hard to do great things, but the empirical evidence is on the scale of the evidence for mortality. That's why it's essential to work on something you're deeply interested in. Interest will drive you to work harder than mere diligence ever could.

The three most powerful motives are curiosity, delight, and the desire to do something impressive. Sometimes they converge, and that combination is the most powerful of all.

The big prize is to discover a new fractal bud. You notice a crack in the surface of knowledge, pry it open, and there's a whole world inside.





Let's talk a little more about the complicated business of figuring out what to work on. The main reason it's hard is that you can't tell what most kinds of work are like except by doing them. Which means the four steps overlap: you may have to work at something for years before you know how much you like it or how good you are at it. And in the meantime you're not doing, and thus not learning about, most other kinds of work. So in the worst case you choose late based on very incomplete information. [4]

The nature of ambition exacerbates this problem. Ambition comes in two forms, one that precedes interest in the subject and one that grows out of it. Most people who do great work have a mix, and the more you have of the former, the harder it will be to decide what to do.

The educational systems in most countries pretend it's easy. They expect you to commit to a field long before you could know what it's really like. And as a result an ambitious person on an optimal trajectory will often read to the system as an instance of breakage.

It would be better if they at least admitted it — if they admitted that the system not only can't do much to help you figure out what to work on, but is designed on the assumption that you'll somehow magically guess as a teenager. They don't tell you, but I will: when it comes to figuring out what to work on, you're on your own. Some people get lucky and do guess correctly, but the rest will find themselves scrambling diagonally across tracks laid down on the assumption that everyone does.

What should you do if you're young and ambitious but don't know what to work on? What you should not do is drift along passively, assuming the problem will solve itself. You need to take action. But there is no systematic procedure you can follow. When you read biographies of people who've done great work, it's remarkable how much luck is involved. They discover what to work on as a result of a chance meeting, or by reading a book they happen to pick up. So you need to make yourself a big target for luck, and the way to do that is to be curious. Try lots of things, meet lots of people, read lots of books, ask lots of questions. [5]

When in doubt, optimize for interestingness. Fields change as you learn more about them. What mathematicians do, for example, is very different from what you do in high school math classes. So you need to give different types of work a chance to show you what they're like. But a field should become increasingly interesting as you learn more about it. If it doesn't, it's probably not for you.

Don't worry if you find you're interested in different things than other people. The stranger your tastes in interestingness, the better. Strange tastes are often strong ones, and a strong taste for work means you'll be productive. And you're more likely to find new things if you're looking where few have looked before.

One sign that you're suited for some kind of work is when you like even the parts that other people find tedious or frightening.

But fields aren't people; you don't owe them any loyalty. If in the course of working on one thing you discover another that's more exciting, don't be afraid to switch.

If you're making something for people, make sure it's something they actually want. The best way to do this is to make something you yourself want. Write the story you want to read; build the tool you want to use. Since your friends probably have similar interests, this will also get you your initial audience.

This should follow from the excitingness rule. Obviously the most exciting story to write will be the one you want to read. The reason I mention this case explicitly is that so many people get it wrong. Instead of making what they want, they try to make what some imaginary, more sophisticated audience wants. And once you go down that route, you're lost. [6]

There are a lot of forces that will lead you astray when you're trying to figure out what to work on. Pretentiousness, fashion, fear, money, politics, other people's wishes, eminent frauds. But if you stick to what you find genuinely interesting, you'll be proof against all of them. If you're interested, you're not astray.





Following your interests may sound like a rather passive strategy, but in practice it usually means following them past all sorts of obstacles. You usually have to risk rejection and failure. So it does take a good deal of boldness.

But while you need boldness, you don't usually need much planning. In most cases the recipe for doing great work is simply: work hard on excitingly ambitious projects, and something good will come of it. Instead of making a plan and then executing it, you just try to preserve certain invariants.

The trouble with planning is that it only works for achievements you can describe in advance. You can win a gold medal or get rich by deciding to as a child and then tenaciously pursuing that goal, but you can't discover natural selection that way.

I think for most people who want to do great work, the right strategy is not to plan too much. At each stage do whatever seems most interesting and gives you the best options for the future. I call this approach "staying upwind." This is how most people who've done great work seem to have done it.





Even when you've found something exciting to work on, working on it is not always straightforward. There will be times when some new idea makes you leap out of bed in the morning and get straight to work. But there will also be plenty of times when things aren't like that.

You don't just put out your sail and get blown forward by inspiration. There are headwinds and currents and hidden shoals. So there's a technique to working, just as there is to sailing.

For example, while you must work hard, it's possible to work too hard, and if you do that you'll find you get diminishing returns: fatigue will make you stupid, and eventually even damage your health. The point at which work yields diminishing returns depends on the type. Some of the hardest types you might only be able to do for four or five hours a day.

Ideally those hours will be contiguous. To the extent you can, try to arrange your life so you have big blocks of time to work in. You'll shy away from hard tasks if you know you might be interrupted.

It will probably be harder to start working than to keep working. You'll often have to trick yourself to get over that initial threshold. Don't worry about this; it's the nature of work, not a flaw in your character. Work has a sort of activation energy, both per day and per project. And since this threshold is fake in the sense that it's higher than the energy required to keep going, it's ok to tell yourself a lie of corresponding magnitude to get over it.

It's usually a mistake to lie to yourself if you want to do great work, but this is one of the rare cases where it isn't. When I'm reluctant to start work in the morning, I often trick myself by saying "I'll just read over what I've got so far." Five minutes later I've found something that seems mistaken or incomplete, and I'm off.

Similar techniques work for starting new projects. It's ok to lie to yourself about how much work a project will entail, for example. Lots of great things began with someone saying "How hard could it be?"

This is one case where the young have an advantage. They're more optimistic, and even though one of the sources of their optimism is ignorance, in this case ignorance can sometimes beat knowledge.

Try to finish what you start, though, even if it turns out to be more work than you expected. Finishing things is not just an exercise in tidiness or self-discipline. In many projects a lot of the best work happens in what was meant to be the final stage.

Another permissible lie is to exaggerate the importance of what you're working on, at least in your own mind. If that helps you discover something new, it may turn out not to have been a lie after all. [7]





Since there are two senses of starting work — per day and per project — there are also two forms of procrastination. Per-project procrastination is far the more dangerous. You put off starting that ambitious project from year to year because the time isn't quite right. When you're procrastinating in units of years, you can get a lot not done. [8]

One reason per-project procrastination is so dangerous is that it usually camouflages itself as work. You're not just sitting around doing nothing; you're working industriously on something else. So per-project procrastination doesn't set off the alarms that per-day procrastination does. You're too busy to notice it.

The way to beat it is to stop occasionally and ask yourself: Am I working on what I most want to work on? When you're young it's ok if the answer is sometimes no, but this gets increasingly dangerous as you get older. [9]





Great work usually entails spending what would seem to most people an unreasonable amount of time on a problem. You can't think of this time as a cost, or it will seem too high. You have to find the work sufficiently engaging as it's happening.

There may be some jobs where you have to work diligently for years at things you hate before you get to the good part, but this is not how great work happens. Great work happens by focusing consistently on something you're genuinely interested in. When you pause to take stock, you're surprised how far you've come.

The reason we're surprised is that we underestimate the cumulative effect of work. Writing a page a day doesn't sound like much, but if you do it every day you'll write a book a year. That's the key: consistency. People who do great things don't get a lot done every day. They get something done, rather than nothing.

If you do work that compounds, you'll get exponential growth. Most people who do this do it unconsciously, but it's worth stopping to think about. Learning, for example, is an instance of this phenomenon: the more you learn about something, the easier it is to learn more. Growing an audience is another: the more fans you have, the more new fans they'll bring you.

The trouble with exponential growth is that the curve feels flat in the beginning. It isn't; it's still a wonderful exponential curve. But we can't grasp that intuitively, so we underrate exponential growth in its early stages.

Something that grows exponentially can become so valuable that it's worth making an extraordinary effort to get it started. But since we underrate exponential growth early on, this too is mostly done unconsciously: people push through the initial, unrewarding phase of learning something new because they know from experience that learning new things always takes an initial push, or they grow their audience one fan at a time because they have nothing better to do. If people consciously realized they could invest in exponential growth, many more would do it.





Work doesn't just happen when you're trying to. There's a kind of undirected thinking you do when walking or taking a shower or lying in bed that can be very powerful. By letting your mind wander a little, you'll often solve problems you were unable to solve by frontal attack.

You have to be working hard in the normal way to benefit from this phenomenon, though. You can't just walk around daydreaming. The daydreaming has to be interleaved with deliberate work that feeds it questions. [10]

Everyone knows to avoid distractions at work, but it's also important to avoid them in the other half of the cycle. When you let your mind wander, it wanders to whatever you care about most at that moment. So avoid the kind of distraction that pushes your work out of the top spot, or you'll waste this valuable type of thinking on the distraction instead. (Exception: Don't avoid love.)





Consciously cultivate your taste in the work done in your field. Until you know which is the best and what makes it so, you don't know what you're aiming for.

And that is what you're aiming for, because if you don't try to be the best, you won't even be good. This observation has been made by so many people in so many different fields that it might be worth thinking about why it's true. It could be because ambition is a phenomenon where almost all the error is in one direction — where almost all the shells that miss the target miss by falling short. Or it could be because ambition to be the best is a qualitatively different thing from ambition to be good. Or maybe being good is simply too vague a standard. Probably all three are true. [11]

Fortunately there's a kind of economy of scale here. Though it might seem like you'd be taking on a heavy burden by trying to be the best, in practice you often end up net ahead. It's exciting, and also strangely liberating. It simplifies things. In some ways it's easier to try to be the best than to try merely to be good.

One way to aim high is to try to make something that people will care about in a hundred years. Not because their opinions matter more than your contemporaries', but because something that still seems good in a hundred years is more likely to be genuinely good.





Don't try to work in a distinctive style. Just try to do the best job you can; you won't be able to help doing it in a distinctive way.

Style is doing things in a distinctive way without trying to. Trying to is affectation.

Affectation is in effect to pretend that someone other than you is doing the work. You adopt an impressive but fake persona, and while you're pleased with the impressiveness, the fakeness is what shows in the work. [12]

The temptation to be someone else is greatest for the young. They often feel like nobodies. But you never need to worry about that problem, because it's self-solving if you work on sufficiently ambitious projects. If you succeed at an ambitious project, you're not a nobody; you're the person who did it. So just do the work and your identity will take care of itself.





"Avoid affectation" is a useful rule so far as it goes, but how would you express this idea positively? How would you say what to be, instead of what not to be? The best answer is earnest. If you're earnest you avoid not just affectation but a whole set of similar vices.

The core of being earnest is being intellectually honest. We're taught as children to be honest as an unselfish virtue — as a kind of sacrifice. But in fact it's a source of power too. To see new ideas, you need an exceptionally sharp eye for the truth. You're trying to see more truth than others have seen so far. And how can you have a sharp eye for the truth if you're intellectually dishonest?

One way to avoid intellectual dishonesty is to maintain a slight positive pressure in the opposite direction. Be aggressively willing to admit that you're mistaken. Once you've admitted you were mistaken about something, you're free. Till then you have to carry it. [13]

Another more subtle component of earnestness is informality. Informality is much more important than its grammatically negative name implies. It's not merely the absence of something. It means focusing on what matters instead of what doesn't.

What formality and affectation have in common is that as well as doing the work, you're trying to seem a certain way as you're doing it. But any energy that goes into how you seem comes out of being good. That's one reason nerds have an advantage in doing great work: they expend little effort on seeming anything. In fact that's basically the definition of a nerd.

Nerds have a kind of innocent boldness that's exactly what you need in doing great work. It's not learned; it's preserved from childhood. So hold onto it. Be the one who puts things out there rather than the one who sits back and offers sophisticated-sounding criticisms of them. "It's easy to criticize" is true in the most literal sense, and the route to great work is never easy.

There may be some jobs where it's an advantage to be cynical and pessimistic, but if you want to do great work it's an advantage to be optimistic, even though that means you'll risk looking like a fool sometimes. There's an old tradition of doing the opposite. The Old Testament says it's better to keep quiet lest you look like a fool. But that's advice for seeming smart. If you actually want to discover new things, it's better to take the risk of telling people your ideas.

Some people are naturally earnest, and with others it takes a conscious effort. Either kind of earnestness will suffice. But I doubt it would be possible to do great work without being earnest. It's so hard to do even if you are. You don't have enough margin for error to accommodate the distortions introduced by being affected, intellectually dishonest, orthodox, fashionable, or cool. [14]





Great work is consistent not only with who did it, but with itself. It's usually all of a piece. So if you face a decision in the middle of working on something, ask which choice is more consistent.

You may have to throw things away and redo them. You won't necessarily have to, but you have to be willing to. And that can take some effort; when there's something you need to redo, status quo bias and laziness will combine to keep you in denial about it. To beat this ask: If I'd already made the change, would I want to revert to what I have now?

Have the confidence to cut. Don't keep something that doesn't fit just because you're proud of it, or because it cost you a lot of effort.

Indeed, in some kinds of work it's good to strip whatever you're doing to its essence. The result will be more concentrated; you'll understand it better; and you won't be able to lie to yourself about whether there's anything real there.

Mathematical elegance may sound like a mere metaphor, drawn from the arts. That's what I thought when I first heard the term "elegant" applied to a proof. But now I suspect it's conceptually prior — that the main ingredient in artistic elegance is mathematical elegance. At any rate it's a useful standard well beyond math.

Elegance can be a long-term bet, though. Laborious solutions will often have more prestige in the short term. They cost a lot of effort and they're hard to understand, both of which impress people, at least temporarily.

Whereas some of the very best work will seem like it took comparatively little effort, because it was in a sense already there. It didn't have to be built, just seen. It's a very good sign when it's hard to say whether you're creating something or discovering it.

When you're doing work that could be seen as either creation or discovery, err on the side of discovery. Try thinking of yourself as a mere conduit through which the ideas take their natural shape.

(Strangely enough, one exception is the problem of choosing a problem to work on. This is usually seen as search, but in the best case it's more like creating something. In the best case you create the field in the process of exploring it.)

Similarly, if you're trying to build a powerful tool, make it gratuitously unrestrictive. A powerful tool almost by definition will be used in ways you didn't expect, so err on the side of eliminating restrictions, even if you don't know what the benefit will be.

Great work will often be tool-like in the sense of being something others build on. So it's a good sign if you're creating ideas that others could use, or exposing questions that others could answer. The best ideas have implications in many different areas.

If you express your ideas in the most general form, they'll be truer than you intended.





True by itself is not enough, of course. Great ideas have to be true and new. And it takes a certain amount of ability to see new ideas even once you've learned enough to get to one of the frontiers of knowledge.

In English we give this ability names like originality, creativity, and imagination. And it seems reasonable to give it a separate name, because it does seem to some extent a separate skill. It's possible to have a great deal of ability in other respects — to have a great deal of what's often called "technical ability" — and yet not have much of this.

I've never liked the term "creative process." It seems misleading. Originality isn't a process, but a habit of mind. Original thinkers throw off new ideas about whatever they focus on, like an angle grinder throwing off sparks. They can't help it.

If the thing they're focused on is something they don't understand very well, these new ideas might not be good. One of the most original thinkers I know decided to focus on dating after he got divorced. He knew roughly as much about dating as the average 15 year old, and the results were spectacularly colorful. But to see originality separated from expertise like that made its nature all the more clear.

I don't know if it's possible to cultivate originality, but there are definitely ways to make the most of however much you have. For example, you're much more likely to have original ideas when you're working on something. Original ideas don't come from trying to have original ideas. They come from trying to build or understand something slightly too difficult. [15]

Talking or writing about the things you're interested in is a good way to generate new ideas. When you try to put ideas into words, a missing idea creates a sort of vacuum that draws it out of you. Indeed, there's a kind of thinking that can only be done by writing.

Changing your context can help. If you visit a new place, you'll often find you have new ideas there. The journey itself often dislodges them. But you may not have to go far to get this benefit. Sometimes it's enough just to go for a walk. [16]

It also helps to travel in topic space. You'll have more new ideas if you explore lots of different topics, partly because it gives the angle grinder more surface area to work on, and partly because analogies are an especially fruitful source of new ideas.

Don't divide your attention evenly between many topics though, or you'll spread yourself too thin. You want to distribute it according to something more like a power law. [17] Be professionally curious about a few topics and idly curious about many more.

Curiosity and originality are closely related. Curiosity feeds originality by giving it new things to work on. But the relationship is closer than that. Curiosity is itself a kind of originality; it's roughly to questions what originality is to answers. And since questions at their best are a big component of answers, curiosity at its best is a creative force.





Having new ideas is a strange game, because it usually consists of seeing things that were right under your nose. Once you've seen a new idea, it tends to seem obvious. Why did no one think of this before?

When an idea seems simultaneously novel and obvious, it's probably a good one.

Seeing something obvious sounds easy. And yet empirically having new ideas is hard. What's the source of this apparent contradiction? It's that seeing the new idea usually requires you to change the way you look at the world. We see the world through models that both help and constrain us. When you fix a broken model, new ideas become obvious. But noticing and fixing a broken model is hard. That's how new ideas can be both obvious and yet hard to discover: they're easy to see after you do something hard.

One way to discover broken models is to be stricter than other people. Broken models of the world leave a trail of clues where they bash against reality. Most people don't want to see these clues. It would be an understatement to say that they're attached to their current model; it's what they think in; so they'll tend to ignore the trail of clues left by its breakage, however conspicuous it may seem in retrospect.

To find new ideas you have to seize on signs of breakage instead of looking away. That's what Einstein did. He was able to see the wild implications of Maxwell's equations not so much because he was looking for new ideas as because he was stricter.

The other thing you need is a willingness to break rules. Paradoxical as it sounds, if you want to fix your model of the world, it helps to be the sort of person who's comfortable breaking rules. From the point of view of the old model, which everyone including you initially shares, the new model usually breaks at least implicit rules.

Few understand the degree of rule-breaking required, because new ideas seem much more conservative once they succeed. They seem perfectly reasonable once you're using the new model of the world they brought with them. But they didn't at the time; it took the greater part of a century for the heliocentric model to be generally accepted, even among astronomers, because it felt so wrong.

Indeed, if you think about it, a good new idea has to seem bad to most people, or someone would have already explored it. So what you're looking for is ideas that seem crazy, but the right kind of crazy. How do you recognize these? You can't with certainty. Often ideas that seem bad are bad. But ideas that are the right kind of crazy tend to be exciting; they're rich in implications; whereas ideas that are merely bad tend to be depressing.

There are two ways to be comfortable breaking rules: to enjoy breaking them, and to be indifferent to them. I call these two cases being aggressively and passively independent-minded.

The aggressively independent-minded are the naughty ones. Rules don't merely fail to stop them; breaking rules gives them additional energy. For this sort of person, delight at the sheer audacity of a project sometimes supplies enough activation energy to get it started.

The other way to break rules is not to care about them, or perhaps even to know they exist. This is why novices and outsiders often make new discoveries; their ignorance of a field's assumptions acts as a source of temporary passive independent-mindedness. Aspies also seem to have a kind of immunity to conventional beliefs. Several I know say that this helps them to have new ideas.

Strictness plus rule-breaking sounds like a strange combination. In popular culture they're opposed. But popular culture has a broken model in this respect. It implicitly assumes that issues are trivial ones, and in trivial matters strictness and rule-breaking are opposed. But in questions that really matter, only rule-breakers can be truly strict.





An overlooked idea often doesn't lose till the semifinals. You do see it, subconsciously, but then another part of your subconscious shoots it down because it would be too weird, too risky, too much work, too controversial. This suggests an exciting possibility: if you could turn off such filters, you could see more new ideas.

One way to do that is to ask what would be good ideas for someone else to explore. Then your subconscious won't shoot them down to protect you.

You could also discover overlooked ideas by working in the other direction: by starting from what's obscuring them. Every cherished but mistaken principle is surrounded by a dead zone of valuable ideas that are unexplored because they contradict it.

Religions are collections of cherished but mistaken principles. So anything that can be described either literally or metaphorically as a religion will have valuable unexplored ideas in its shadow. Copernicus and Darwin both made discoveries of this type. [18]

What are people in your field religious about, in the sense of being too attached to some principle that might not be as self-evident as they think? What becomes possible if you discard it?





People show much more originality in solving problems than in deciding which problems to solve. Even the smartest can be surprisingly conservative when deciding what to work on. People who'd never dream of being fashionable in any other way get sucked into working on fashionable problems.

One reason people are more conservative when choosing problems than solutions is that problems are bigger bets. A problem could occupy you for years, while exploring a solution might only take days. But even so I think most people are too conservative. They're not merely responding to risk, but to fashion as well. Unfashionable problems are undervalued.

One of the most interesting kinds of unfashionable problem is the problem that people think has been fully explored, but hasn't. Great work often takes something that already exists and shows its latent potential. Durer and Watt both did this. So if you're interested in a field that others think is tapped out, don't let their skepticism deter you. People are often wrong about this.

Working on an unfashionable problem can be very pleasing. There's no hype or hurry. Opportunists and critics are both occupied elsewhere. The existing work often has an old-school solidity. And there's a satisfying sense of economy in cultivating ideas that would otherwise be wasted.

But the most common type of overlooked problem is not explicitly unfashionable in the sense of being out of fashion. It just doesn't seem to matter as much as it actually does. How do you find these? By being self-indulgent — by letting your curiosity have its way, and tuning out, at least temporarily, the little voice in your head that says you should only be working on "important" problems.

You do need to work on important problems, but almost everyone is too conservative about what counts as one. And if there's an important but overlooked problem in your neighborhood, it's probably already on your subconscious radar screen. So try asking yourself: if you were going to take a break from "serious" work to work on something just because it would be really interesting, what would you do? The answer is probably more important than it seems.

Originality in choosing problems seems to matter even more than originality in solving them. That's what distinguishes the people who discover whole new fields. So what might seem to be merely the initial step — deciding what to work on — is in a sense the key to the whole game.





Few grasp this. One of the biggest misconceptions about new ideas is about the ratio of question to answer in their composition. People think big ideas are answers, but often the real insight was in the question.

Part of the reason we underrate questions is the way they're used in schools. In schools they tend to exist only briefly before being answered, like unstable particles. But a really good question can be much more than that. A really good question is a partial discovery. How do new species arise? Is the force that makes objects fall to earth the same as the one that keeps planets in their orbits? By even asking such questions you were already in excitingly novel territory.

Unanswered questions can be uncomfortable things to carry around with you. But the more you're carrying, the greater the chance of noticing a solution — or perhaps even more excitingly, noticing that two unanswered questions are the same.

Sometimes you carry a question for a long time. Great work often comes from returning to a question you first noticed years before — in your childhood, even — and couldn't stop thinking about. People talk a lot about the importance of keeping your youthful dreams alive, but it's just as important to keep your youthful questions alive. [19]

This is one of the places where actual expertise differs most from the popular picture of it. In the popular picture, experts are certain. But actually the more puzzled you are, the better, so long as (a) the things you're puzzled about matter, and (b) no one else understands them either.

Think about what's happening at the moment just before a new idea is discovered. Often someone with sufficient expertise is puzzled about something. Which means that originality consists partly of puzzlement — of confusion! You have to be comfortable enough with the world being full of puzzles that you're willing to see them, but not so comfortable that you don't want to solve them. [20]

It's a great thing to be rich in unanswered questions. And this is one of those situations where the rich get richer, because the best way to acquire new questions is to try answering existing ones. Questions don't just lead to answers, but also to more questions.





The best questions grow in the answering. You notice a thread protruding from the current paradigm and try pulling on it, and it just gets longer and longer. So don't require a question to be obviously big before you try answering it. You can rarely predict that. It's hard enough even to notice the thread, let alone to predict how much will unravel if you pull on it.

It's better to be promiscuously curious — to pull a little bit on a lot of threads, and see what happens. Big things start small. The initial versions of big things were often just experiments, or side projects, or talks, which then grew into something bigger. So start lots of small things.

Being prolific is underrated. The more different things you try, the greater the chance of discovering something new. Understand, though, that trying lots of things will mean trying lots of things that don't work. You can't have a lot of good ideas without also having a lot of bad ones. [21]

Though it sounds more responsible to begin by studying everything that's been done before, you'll learn faster and have more fun by trying stuff. And you'll understand previous work better when you do look at it. So err on the side of starting. Which is easier when starting means starting small; those two ideas fit together like two puzzle pieces.

How do you get from starting small to doing something great? By making successive versions. Great things are almost always made in successive versions. You start with something small and evolve it, and the final version is both cleverer and more ambitious than anything you could have planned.

It's particularly useful to make successive versions when you're making something for people — to get an initial version in front of them quickly, and then evolve it based on their response.

Begin by trying the simplest thing that could possibly work. Surprisingly often, it does. If it doesn't, this will at least get you started.

Don't try to cram too much new stuff into any one version. There are names for doing this with the first version (taking too long to ship) and the second (the second system effect), but these are both merely instances of a more general principle.

An early version of a new project will sometimes be dismissed as a toy. It's a good sign when people do this. That means it has everything a new idea needs except scale, and that tends to follow. [22]

The alternative to starting with something small and evolving it is to plan in advance what you're going to do. And planning does usually seem the more responsible choice. It sounds more organized to say "we're going to do x and then y and then z" than "we're going to try x and see what happens." And it is more organized; it just doesn't work as well.

Planning per se isn't good. It's sometimes necessary, but it's a necessary evil — a response to unforgiving conditions. It's something you have to do because you're working with inflexible media, or because you need to coordinate the efforts of a lot of people. If you keep projects small and use flexible media, you don't have to plan as much, and your designs can evolve instead.





Take as much risk as you can afford. In an efficient market, risk is proportionate to reward, so don't look for certainty, but for a bet with high expected value. If you're not failing occasionally, you're probably being too conservative.

Though conservatism is usually associated with the old, it's the young who tend to make this mistake. Inexperience makes them fear risk, but it's when you're young that you can afford the most.

Even a project that fails can be valuable. In the process of working on it, you'll have crossed territory few others have seen, and encountered questions few others have asked. And there's probably no better source of questions than the ones you encounter in trying to do something slightly too hard.





Use the advantages of youth when you have them, and the advantages of age once you have those. The advantages of youth are energy, time, optimism, and freedom. The advantages of age are knowledge, efficiency, money, and power. With effort you can acquire some of the latter when young and keep some of the former when old.

The old also have the advantage of knowing which advantages they have. The young often have them without realizing it. The biggest is probably time. The young have no idea how rich they are in time. The best way to turn this time to advantage is to use it in slightly frivolous ways: to learn about something you don't need to know about, just out of curiosity, or to try building something just because it would be cool, or to become freakishly good at something.

That "slightly" is an important qualification. Spend time lavishly when you're young, but don't simply waste it. There's a big difference between doing something you worry might be a waste of time and doing something you know for sure will be. The former is at least a bet, and possibly a better one than you think. [23]

The most subtle advantage of youth, or more precisely of inexperience, is that you're seeing everything with fresh eyes. When your brain embraces an idea for the first time, sometimes the two don't fit together perfectly. Usually the problem is with your brain, but occasionally it's with the idea. A piece of it sticks out awkwardly and jabs you when you think about it. People who are used to the idea have learned to ignore it, but you have the opportunity not to. [24]

So when you're learning about something for the first time, pay attention to things that seem wrong or missing. You'll be tempted to ignore them, since there's a 99% chance the problem is with you. And you may have to set aside your misgivings temporarily to keep progressing. But don't forget about them. When you've gotten further into the subject, come back and check if they're still there. If they're still viable in the light of your present knowledge, they probably represent an undiscovered idea.





One of the most valuable kinds of knowledge you get from experience is to know what you don't have to worry about. The young know all the things that could matter, but not their relative importance. So they worry equally about everything, when they should worry much more about a few things and hardly at all about the rest.

But what you don't know is only half the problem with inexperience. The other half is what you do know that ain't so. You arrive at adulthood with your head full of nonsense — bad habits you've acquired and false things you've been taught — and you won't be able to do great work till you clear away at least the nonsense in the way of whatever type of work you want to do.

Much of the nonsense left in your head is left there by schools. We're so used to schools that we unconsciously treat going to school as identical with learning, but in fact schools have all sorts of strange qualities that warp our ideas about learning and thinking.

For example, schools induce passivity. Since you were a small child, there was an authority at the front of the class telling all of you what you had to learn and then measuring whether you did. But neither classes nor tests are intrinsic to learning; they're just artifacts of the way schools are usually designed.

The sooner you overcome this passivity, the better. If you're still in school, try thinking of your education as your project, and your teachers as working for you rather than vice versa. That may seem a stretch, but it's not merely some weird thought experiment. It's the truth, economically, and in the best case it's the truth intellectually as well. The best teachers don't want to be your bosses. They'd prefer it if you pushed ahead, using them as a source of advice, rather than being pulled by them through the material.

Schools also give you a misleading impression of what work is like. In school they tell you what the problems are, and they're almost always soluble using no more than you've been taught so far. In real life you have to figure out what the problems are, and you often don't know if they're soluble at all.

But perhaps the worst thing schools do to you is train you to win by hacking the test. You can't do great work by doing that. You can't trick God. So stop looking for that kind of shortcut. The way to beat the system is to focus on problems and solutions that others have overlooked, not to skimp on the work itself.





Don't think of yourself as dependent on some gatekeeper giving you a "big break." Even if this were true, the best way to get it would be to focus on doing good work rather than chasing influential people.

And don't take rejection by committees to heart. The qualities that impress admissions officers and prize committees are quite different from those required to do great work. The decisions of selection committees are only meaningful to the extent that they're part of a feedback loop, and very few are.





People new to a field will often copy existing work. There's nothing inherently bad about that. There's no better way to learn how something works than by trying to reproduce it. Nor does copying necessarily make your work unoriginal. Originality is the presence of new ideas, not the absence of old ones.

There's a good way to copy and a bad way. If you're going to copy something, do it openly instead of furtively, or worse still, unconsciously. This is what's meant by the famously misattributed phrase "Great artists steal." The really dangerous kind of copying, the kind that gives copying a bad name, is the kind that's done without realizing it, because you're nothing more than a train running on tracks laid down by someone else. But at the other extreme, copying can be a sign of superiority rather than subordination. [25]

In many fields it's almost inevitable that your early work will be in some sense based on other people's. Projects rarely arise in a vacuum. They're usually a reaction to previous work. When you're first starting out, you don't have any previous work; if you're going to react to something, it has to be someone else's. Once you're established, you can react to your own. But while the former gets called derivative and the latter doesn't, structurally the two cases are more similar than they seem.

Oddly enough, the very novelty of the most novel ideas sometimes makes them seem at first to be more derivative than they are. New discoveries often have to be conceived initially as variations of existing things, even by their discoverers, because there isn't yet the conceptual vocabulary to express them.

There are definitely some dangers to copying, though. One is that you'll tend to copy old things — things that were in their day at the frontier of knowledge, but no longer are.

And when you do copy something, don't copy every feature of it. Some will make you ridiculous if you do. Don't copy the manner of an eminent 50 year old professor if you're 18, for example, or the idiom of a Renaissance poem hundreds of years later.

Some of the features of things you admire are flaws they succeeded despite. Indeed, the features that are easiest to imitate are the most likely to be the flaws.

This is particularly true for behavior. Some talented people are jerks, and this sometimes makes it seem to the inexperienced that being a jerk is part of being talented. It isn't; being talented is merely how they get away with it.

One of the most powerful kinds of copying is to copy something from one field into another. History is so full of chance discoveries of this type that it's probably worth giving chance a hand by deliberately learning about other kinds of work. You can take ideas from quite distant fields if you let them be metaphors.

Negative examples can be as inspiring as positive ones. In fact you can sometimes learn more from things done badly than from things done well; sometimes it only becomes clear what's needed when it's missing.





If a lot of the best people in your field are collected in one place, it's usually a good idea to visit for a while. It will increase your ambition, and also, by showing you that these people are human, increase your self-confidence. [26]

If you're earnest you'll probably get a warmer welcome than you might expect. Most people who are very good at something are happy to talk about it with anyone who's genuinely interested. If they're really good at their work, then they probably have a hobbyist's interest in it, and hobbyists always want to talk about their hobbies.

It may take some effort to find the people who are really good, though. Doing great work has such prestige that in some places, particularly universities, there's a polite fiction that everyone is engaged in it. And that is far from true. People within universities can't say so openly, but the quality of the work being done in different departments varies immensely. Some departments have people doing great work; others have in the past; others never have.





Seek out the best colleagues. There are a lot of projects that can't be done alone, and even if you're working on one that can be, it's good to have other people to encourage you and to bounce ideas off.

Colleagues don't just affect your work, though; they also affect you. So work with people you want to become like, because you will.

Quality is more important than quantity in colleagues. It's better to have one or two great ones than a building full of pretty good ones. In fact it's not merely better, but necessary, judging from history: the degree to which great work happens in clusters suggests that one's colleagues often make the difference between doing great work and not.

How do you know when you have sufficiently good colleagues? In my experience, when you do, you know. Which means if you're unsure, you probably don't. But it may be possible to give a more concrete answer than that. Here's an attempt: sufficiently good colleagues offer surprising insights. They can see and do things that you can't. So if you have a handful of colleagues good enough to keep you on your toes in this sense, you're probably over the threshold.

Most of us can benefit from collaborating with colleagues, but some projects require people on a larger scale, and starting one of those is not for everyone. If you want to run a project like that, you'll have to become a manager, and managing well takes aptitude and interest like any other kind of work. If you don't have them, there is no middle path: you must either force yourself to learn management as a second language, or avoid such projects. [27]





Husband your morale. It's the basis of everything when you're working on ambitious projects. You have to nurture and protect it like a living organism.

Morale starts with your view of life. You're more likely to do great work if you're an optimist, and more likely to if you think of yourself as lucky than if you think of yourself as a victim.

Indeed, work can to some extent protect you from your problems. If you choose work that's pure, its very difficulties will serve as a refuge from the difficulties of everyday life. If this is escapism, it's a very productive form of it, and one that has been used by some of the greatest minds in history.

Morale compounds via work: high morale helps you do good work, which increases your morale and helps you do even better work. But this cycle also operates in the other direction: if you're not doing good work, that can demoralize you and make it even harder to. Since it matters so much for this cycle to be running in the right direction, it can be a good idea to switch to easier work when you're stuck, just so you start to get something done.

One of the biggest mistakes ambitious people make is to allow setbacks to destroy their morale all at once, like a balloon bursting. You can inoculate yourself against this by explicitly considering setbacks a part of your process. Solving hard problems always involves some backtracking.

Doing great work is a depth-first search whose root node is the desire to. So "If at first you don't succeed, try, try again" isn't quite right. It should be: If at first you don't succeed, either try again, or backtrack and then try again.

"Never give up" is also not quite right. Obviously there are times when it's the right choice to eject. A more precise version would be: Never let setbacks panic you into backtracking more than you need to. Corollary: Never abandon the root node.

It's not necessarily a bad sign if work is a struggle, any more than it's a bad sign to be out of breath while running. It depends how fast you're running. So learn to distinguish good pain from bad. Good pain is a sign of effort; bad pain is a sign of damage.





An audience is a critical component of morale. If you're a scholar, your audience may be your peers; in the arts, it may be an audience in the traditional sense. Either way it doesn't need to be big. The value of an audience doesn't grow anything like linearly with its size. Which is bad news if you're famous, but good news if you're just starting out, because it means a small but dedicated audience can be enough to sustain you. If a handful of people genuinely love what you're doing, that's enough.

To the extent you can, avoid letting intermediaries come between you and your audience. In some types of work this is inevitable, but it's so liberating to escape it that you might be better off switching to an adjacent type if that will let you go direct. [28]

The people you spend time with will also have a big effect on your morale. You'll find there are some who increase your energy and others who decrease it, and the effect someone has is not always what you'd expect. Seek out the people who increase your energy and avoid those who decrease it. Though of course if there's someone you need to take care of, that takes precedence.

Don't marry someone who doesn't understand that you need to work, or sees your work as competition for your attention. If you're ambitious, you need to work; it's almost like a medical condition; so someone who won't let you work either doesn't understand you, or does and doesn't care.

Ultimately morale is physical. You think with your body, so it's important to take care of it. That means exercising regularly, eating and sleeping well, and avoiding the more dangerous kinds of drugs. Running and walking are particularly good forms of exercise because they're good for thinking. [29]

People who do great work are not necessarily happier than everyone else, but they're happier than they'd be if they didn't. In fact, if you're smart and ambitious, it's dangerous not to be productive. People who are smart and ambitious but don't achieve much tend to become bitter.





It's ok to want to impress other people, but choose the right people. The opinion of people you respect is signal. Fame, which is the opinion of a much larger group you might or might not respect, just adds noise.

The prestige of a type of work is at best a trailing indicator and sometimes completely mistaken. If you do anything well enough, you'll make it prestigious. So the question to ask about a type of work is not how much prestige it has, but how well it could be done.

Competition can be an effective motivator, but don't let it choose the problem for you; don't let yourself get drawn into chasing something just because others are. In fact, don't let competitors make you do anything much more specific than work harder.

Curiosity is the best guide. Your curiosity never lies, and it knows more than you do about what's worth paying attention to.





Notice how often that word has come up. If you asked an oracle the secret to doing great work and the oracle replied with a single word, my bet would be on "curiosity."

That doesn't translate directly to advice. It's not enough just to be curious, and you can't command curiosity anyway. But you can nurture it and let it drive you.

Curiosity is the key to all four steps in doing great work: it will choose the field for you, get you to the frontier, cause you to notice the gaps in it, and drive you to explore them. The whole process is a kind of dance with curiosity.





Believe it or not, I tried to make this essay as short as I could. But its length at least means it acts as a filter. If you made it this far, you must be interested in doing great work. And if so you're already further along than you might realize, because the set of people willing to want to is small.

The factors in doing great work are factors in the literal, mathematical sense, and they are: ability, interest, effort, and luck. Luck by definition you can't do anything about, so we can ignore that. And we can assume effort, if you do in fact want to do great work. So the problem boils down to ability and interest. Can you find a kind of work where your ability and interest will combine to yield an explosion of new ideas?

Here there are grounds for optimism. There are so many different ways to do great work, and even more that are still undiscovered. Out of all those different types of work, the one you're most suited for is probably a pretty close match. Probably a comically close match. It's just a question of finding it, and how far into it your ability and interest can take you. And you can only answer that by trying.

Many more people could try to do great work than do. What holds them back is a combination of modesty and fear. It seems presumptuous to try to be Newton or Shakespeare. It also seems hard; surely if you tried something like that, you'd fail. Presumably the calculation is rarely explicit. Few people consciously decide not to try to do great work. But that's what's going on subconsciously; they shy away from the question.

So I'm going to pull a sneaky trick on you. Do you want to do great work, or not? Now you have to decide consciously. Sorry about that. I wouldn't have done it to a general audience. But we already know you're interested.

Don't worry about being presumptuous. You don't have to tell anyone. And if it's too hard and you fail, so what? Lots of people have worse problems than that. In fact you'll be lucky if it's the worst problem you have.

Yes, you'll have to work hard. But again, lots of people have to work hard. And if you're working on something you find very interesting, which you necessarily will if you're on the right path, the work will probably feel less burdensome than a lot of your peers'.

The discoveries are out there, waiting to be made. Why not by you?



"""


pg=pg.split('.')
rh=rh.split('.')
#above .7 = semantically similar
#below 0 = different subject different verb
#below .3 similar subject
#cs=computeSimilarity(pg, rh)


# pge = encode(pg)
# rhe = encode(rh)


# In[ ]:





sorted_pge, indices = pge.sort()
sorted_rhe,indices =  rhe.sort()

#use indices


# In[ ]:


pairs

def display_pair(pair):
    i, j = pair['index']
    print("{} \t\t {} \t\t Score: {:.4f}".format(rh[i], pg[j], pair['score']))
    #print("{} \t\t  \t\t Score: {:.4f}".format(pg[j] if len(rh[i]) > len(pg[j]) else rh[i], pair['score']))


for i in range(100):
    display_pair(pairs[i])
    
    
#you know they are similar enough -> how do you merge - merge by relevance


# In[ ]:


def computeSimilarity(s1, s2):
    return util.cos_sim(s1, s2)
computeSimilarity(sorted_pge, sorted_rhe[:664])


# In[ ]:


p1 = 'i went to the grocery store. I found a box of tomatoes. it was cool.'

p2 = 'i went to the grocery store. i dropped a box of tomatoes. it was cool.'

computeSimilarity(p1, p2)


#along with similarity -> compute how they are different 


# In[ ]:


one= 'i went to the mall with my cousins and ate subway'

two = 'my cousins and i ate subway at the mall.'


# In[ ]:


pos_tag(word_tokenize(one)), pos_tag(word_tokenize(two)) 


# In[ ]:


computeSimilarity('ate soup', 'love soup')


# In[ ]:


from sentence_transformers import SentenceTransformer,util

computeSimilarity('ate a carrot', 'ate some carrots')


# In[ ]:



from sentence_transformers import SentenceTransformer,util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
person1 = 'i want the government to add a railroad between montreal and kansas'
person5 = 'i want the government to add a railroad between kansas and montreal'
person2 = 'i want to build a registry for synthetic plants'


# In[ ]:


model.encode([person1, person2], convert_to_tensor=True)


# In[ ]:


import nltk

trythisone = 'why do you call it an xbox 360? because when you see it, you turn 360 degrees and walk away.'

data = """I used to play piano by ear, but now I use my hands.

I'm reading a book about anti-gravity, and it's impossible to put down.

The math book looked sad, so I asked what's the problem. It said it had too many problems.

A bicycle can't stand on its own because it's two-tired.

I told my friend 10 jokes to make him laugh, but no pun in ten did.

Did you hear about the guy who lost his left side? He's all right now.

I used to be a baker, but I couldn't make enough dough.

When a clock is hungry, it goes back four seconds.

I got a job at a bakery because I kneaded the dough.

I'm friends with all electricians. We have good current connections.""".split('\n')

data2 = """
How do you throw a space party? You planet.

How was Rome split in two? With a pair of Ceasars.

Nope. Unintended.

The shovel was a ground breaking invention, but everyone was blow away by the leaf blower.

A scarecrow says, "This job isn't for everyone, but hay, it's in my jeans."

A Buddhist walks up to a hot dog stand and says "Make me one with everything."

Did you hear about the guy who lost the left side of his body? He's alright now.

What do you call a girl with one leg that's shorter than the other? Ilene.

The broom swept the nation away.

I did a theatrical performance on puns. It was a play on words.

What does a clock do when it's hungry? It goes back for seconds.

What do you do with a dead chemist? You barium.

I bet the person who created the door knocker won a Nobel prize.

Towels can’t tell jokes. They have a dry sense of humor.

Two birds are sitting on a perch and one says “Do you smell fish?”

Did you hear about the cheese factory that exploded in france? There was nothing but des brie.

Do you know sign language? You should learn it, it’s pretty handy.

What do you call a beautiful pumpkin? GOURDgeous.

Why did one banana spy on the other? Because she was appealing.

What do you call a cow with no legs? Ground beef.

What do you call a cow with two legs? Lean beef.

What do you call a cow with all of its legs? High steaks.

A cross eyed teacher couldn’t control his pupils.

After the accident, the juggler didn’t have the balls to do it.

I used to be afraid of hurdles, but I got over it.

To write with a broken pencil is pointless.

I read a book on anti-gravity. I couldn’t put it down.

I couldn’t remember how to throw a boomerang but it came back to me.

What did the buffalo say to his son? Bison.

What should you do if you’re cold? Stand in the corner. It’s 90 degrees.

How does Moses make coffee? Hebrews it.

The energizer bunny went to jail. He was charged with battery.

What did the alien say to the pitcher of water? Take me to your liter.

What happens when you eat too many spaghettiOs? You have a vowel movement.

The soldier who survived mustard gas and pepper spray was a seasoned veteran.

Sausage puns are the wurst.

What do you call a bear with no teeth? A gummy bear.

How did Darth Vader know what luke was getting him for his birthday? He could sense his presence.

Why shouldn’t you trust atoms? They make up everything.

What’s the difference between a bench, a fish, and a bucket of glue? You can’t tune a bench but you can tuna fish. I bet you got stuck on the bucket of glue part.

What’s it called when you have too many aliens? Extraterrestrials.

Want to hear a pizza joke? Nevermind, it’s too cheesy.

What do you call a fake noodle? An impasta.

What do cows tell each other at bedtime? Dairy tales.

Why can’t you take inventory in Afghanistan? Because of the tally ban.

Why didn’t the lion win the race? Because he was racing a cheetah.

Why did the man dig a hole in his neighbor’s backyard and fill it with water? Because he meant well.

What happens to nitrogen when the sun comes up? It becomes daytrogen.

What’s it called when you put a cow in an elevator? Raising the steaks.

What’s america’s favorite soda? Mini soda.

Why did the tomato turn red? Because it saw the salad dressing.

What kind of car does a sheep drive? A lamborghini, but if that breaks down they drive their SuBAHHru.

What do you call a spanish pig? Porque.

What do you call a line of rabbits marching backwards? A receding hairline.

Why don’t vampires go to barbecues? They don’t like steak.

A cabbage and celery walk into a bar and the cabbage gets served first because he was a head.

How do trees access the internet? They log on.

Why should you never trust a train? They have loco motives.

""".split('\n')

import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def isPun(one, two):
    return two in prouncing.rhymes(one)

def isPunSentence(sentence):
    tokens = sentence.split(' ')
    hasNotFoundRhyme = False
    for i in tokens:
        for j in tokens:
            if i ==j: continue
            if isPun(one, two): hasNotFoundRhyme = True
    return hasNotFoundRhyme


data3 = 'What do you call a fake noodle? An impasta.'

[correction(data) for data in data3.split(' ')]
correction('impasta')
#get the original source code to humor 

#anaylze humor + scene understaning in every transcript in modern family, seinfeld, parks+rec, south park

#train LLM from most upvoted comments from hacker news, reddit and so on -> give tptacek+ patio11 personality + etc

fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")


# In[ ]:



def summarize(text) :
    summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
    return summarizer(text)

summarize('im going to go for a walk and type type tpye until text editor is cooler than pasta')

