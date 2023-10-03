from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
import uvicorn
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
import torch
import json
import os.path
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('words')
import random


# convert sentences to encoding
# find similarity between encodings
#user sends sentence
#have premade sentences for each code snippet

def is_real_word(word):
    return word.lower() in nltk.corpus.words.words()

def processMessages(content):
    content = [replaceIfInTags(item) for sublist in content for item in sublist]
    content = [getClassification(string) for string in content 
               if len([w for w in word_tokenize(string) if is_real_word(w)]) > 0]
    misc = [string for string in content
            if len([w for w in word_tokenize(string) if is_real_word(w)]) == 0
    ]
    result = getSimilarity(content)

    returnValue = {}
    for grouping in result:
        title = grouping[0].split(':')[1]
        if len(title) < 1: title = 'unknown'
        returnValue[title] = grouping
    return returnValue


def getClassification(string):
    p = int(random.random() * 5)
    nouns = findNouns(string)
    verb_most_acted_on = nouns #findNouns(string)[0] if len(nouns) > 0 else ''
    return f'{classifications[p]}:  {" ".join(verb_most_acted_on)}'

def processTag(tagged_sentence):
    return [(orig,tag_map[actual_tag]) for (orig,actual_tag) in tagged_sentence if actual_tag in tag_map]

def findNouns(string):
    return [noun for noun,tag in processTag(pos_tag(word_tokenize(string))) ]   
