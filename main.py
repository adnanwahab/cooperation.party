import json
from compiled_functions import jupyter_functions
import glob
import asyncio
import inspect
from shapely.geometry import shape, Point
import random 
import torch
import requests
import easyocr
from fastapi import Request, FastAPI
import random
import json 
import subprocess
import json
import torch
import youtube_dl
import openai
import re
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Optional
import json
import os.path

def removeWhiteSpace(_):
    return _.strip()

access_token = 'pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjI1MmNheTBkZmcyeGwwNWRnZmxxMzEifQ.7KOCTZCiV4QfSeqCQl7HjA'
__ = {}
def initAlgae():
    model_name_or_path = "TheBloke/CodeLlama-7B-Python-GPTQ"
    __['___'] = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                revision="main")
    __['____'] = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
def makeFunctionFromText(text):
    if text == '': return ''
    if '___' not in  __: initAlgae()
    prompt = "sum all numbers from 1 to 10,000"
    prompt_template=f'''[INST] Write a code in javascript to sum fibonacci from 1 to 100```:
    {prompt}
    [/INST]
    '''
    input_ids = __['____'](prompt_template, return_tensors='pt').input_ids.cuda()
    output = __['___'].generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    return re.match(r'[SOL](.*)[/SOL]', __['____'].decode(output[0]))

def makeFnFromEnglish(english):
    fnText = makeFunctionFromText(english)
    return fnText

def is_real_word(word):
    word_list = words.words()
    return word.lower() in word_list

#todo - make reactive -> when parameters change, recompute -> update anything that reads
fn_cache = {}

def cacheThisFunction(func):
    def _(*args):
        key = func.__name__ + args.join(',')
        if key in fn_cache: return fn_cache[key]
        val = func(*args)
        fn_cache[key] = val
    return _




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
def getEncodings(sentences):
    from sentence_transformers import SentenceTransformer,util
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(sentences, convert_to_tensor=True, device='cpu')


def computeSimilarity(s1, s2):
    if (s1 == s2): return False
    embedding_1= model.encode(s1, convert_to_tensor=True)
    embedding_2 = model.encode(s2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embedding_1, embedding_2)
    return sim < .75

def getSimilarity(sentences):
    encodings = getEncodings(sentences)
    clusters = util.community_detection(encodings, min_community_size=1, threshold=0.55)
    def process(item): return [sentences[i] for i in item]
    result = [process(item) for item in clusters ]
    return result

def replaceIfInTags(string):
    for tag in tags:
        if tag in string:
            string = string.replace(tag, matchTags(string))
    return string

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

import random
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://cooperation.party",
    "http://cooperation.party",
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://merge-sentences-todo.ngrok.io/",
    "https://merge-sentences-todo.ngrok.io/",
    "*",
    "pypypy.ngrok.io",
    "http://localhost:5173",
    "localhost:5173"  
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

messages = []
cells = {}
count = 0
locks_mutexes = defaultdict(bool)

class MessageInput(BaseModel):
    text: list[str]
    cellName: int

@app.post("/sendmessage")
async def receive_message(message: MessageInput):
    text = message.text
    cellName = message.cellName
    cells[cellName] = [text for text in text if len(text) > 0]
    messages = [cell for cell in list(cells.values()) if len(cell) > 0]
    if (len(messages) < 1): return JSONResponse(content=[])
    with open('database.txt', 'w') as db: json.dump(result, db)
    return JSONResponse(content=message)


@app.post('/concatIntersection')
async def concat_messages(message: MessageInput):
    cellName = message.cellName
    text = message.text
    cells[cellName] = [text for text in text if len(text) > 0]
    messages = [cell for cell in list(cells.values()) if len(cell) > 0]
    if (len(messages) < 1): return JSONResponse(content=[])
    return content

class RPCBlahBlah(BaseModel):
    text: str

@app.post("/rpc")
async def rpc(message: RPCBlahBlah):
    return {'demo': exec(message.text)}

@app.get("/nexus")
async def index():
    return JSONResponse(content={'456':123})

def getProgram(_, sentence):
    encodings = getEncodings(sentence)
    program_generator_cache = json.load(open('encodings.json', 'w'))
    if encodings in program_generator_cache: return program_generator_cache[encodings]

    json.dump(program_generator_cache, open('encodings.json', 'w'))
    return {'fn': program_generator_cache[encodings]}

def generateWinningTeam():
    from ipynb.fs.defs.geospatial import getCounter
    return getCounter('celebi')

def findAirbnb(previous, sentence):
    from ipynb.fs.defs.geospatial import getAllAirbnbInCityThatAreNotNoisy
    GTorLT = 'not noisy' in sentence
    data = getAllAirbnbInCityThatAreNotNoisy(GTorLT) #todo make reactive live query
    return data

import os
def url_to_file_name(url):
    return re.sub(r'[^a-zA-Z0-9]', '_', url)


import re

def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None



import geopy.distance
 
def geoDistance(one, two):
    return geopy.distance.geodesic(one, two).km

def getPlacesOfInterest(aptGeoLocation):
    aptGeoLocation = aptGeoLocation.split(':')
    aptGeoLocation =  [float(aptGeoLocation[0]), float(aptGeoLocation[1])]
    all_json = []
    return 0
    if not aptGeoLocation: return print('no aptGeoLocation')
    latitude = aptGeoLocation[1]
    longitude = aptGeoLocation[0]
    url = f"""https://api.mapbox.com/search/searchbox/v1/category/shopping?access_token=pk.eyJ1Ijoic2VhcmNoLW1hY2hpbmUtdXNlci0xIiwiYSI6ImNrNnJ6bDdzdzA5cnAza3F4aTVwcWxqdWEifQ.RFF7CVFKrUsZVrJsFzhRvQ&language=en&limit=20&proximity={longitude}%2C%20{latitude}"""
    _ = requests.get(url).json()
    if 'features' not in _: 
        print(_)
        return 0
    for place in _['features']:
        #print(place)
        all_json.append(place)
    poi = []
    for place in all_json:
        coords = place['geometry']['coordinates']
        categories = place['properties']['poi_category']
        poi.append([coords, categories])
        #print(place)
    sorted(poi, key=lambda _: geoDistance(_[0], aptGeoLocation))
    print(poi)
    return geoDistance(poi[0][0], aptGeoLocation)


cache = {}

def addToCache(fn, **kwargs):
    #fn(**kwargs)
    #cache[f'{fn.__name__}:city:aptUrl'] = result
    return fn(**kwargs)


def my_decorator_func(func):
    def wrapper_func(*args, **kwargs):
        # Do something before the function.
        result = func(*args, **kwargs)
        result = addToCache(func, **kwargs)
        #saveCacheToDiskOrRedisOrSqlLiteOr?
        #   
        # Do something after the function.
    return wrapper_func

#get ARIBNBS -> 
#convert AIRBNS to CITY NAME 

    #isochroneLibrary()
    # return [apt for apt in airbnbs]



    # distance_to_shopping_store = [getPlacesOfInterest(geoCoordinate[0]) for geoCoordinate in geoCoordinates
    #                               if len(geoCoordinate) > 0
                                  
    # for every apt           
    #   get geojson 
    #   get all cafes within bounding box of isochrone
    #   then do point in polygon intersection
    #   if cafes within isochrone > 1-> return true
    #get isochrone for subway 
    #get places from google for best completenes
    # #Google Places API   
    #points = [fetch_cafes() for ]
    
    #return [{'link': 'asdfasd'}, {'link': str(len(geoCoordinates))}]
    #print(distance_to_shopping_store)
    #print('this is cool',len(distance_to_shopping_store), len(airbnbs))
    #get 10 airbnb that are closest to shopping out of list in 20 airbnbs
    #tenth = distance_to_shopping_store.copy().sort()
    #make instant by caching all places and all geo-coordinate.
    #what are pros/cons of 1 list per column vs 10 columns per item?
    #how to add 10 "synthetic" or custom columns based on user input in a generic way?
    return [apt for idx, apt in enumerate(airbnbs)
            #if distance_to_shopping_store[idx] < .1
            ]

def landDistribution(_, sentence):
    #within 10 mile commute time
    #

    return 123
    #return landDistribution()

def trees_map(_, sentence):
    return {
        'data': [[34, 34], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        'component': '<map>'
    }

import requests

async def delay(time):
    await asyncio.sleep(time)
import pyppeteer
async def get_html(channel_name):
    browser = await pyppeteer.launch(headless=True)
    page = await browser.newPage()

    await page.setViewport({"width": 1920, "height": 1080})
    await page.goto(f"https://www.twitch.tv/{channel_name}", {"waitUntil": "networkidle2"})

    await delay(1)

    await page.waitForFunction("""
    () => {
        const el = document.querySelector('div[data-a-target="chat-welcome-message"].chat-line__status');
        return !!el;
    }
    """, polling="raf")

    selector = '.chat-line__message, .chat-line__status, div[data-a-target="chat-line-message"]'
    await page.waitForSelector(selector)

    text_elements = await page.querySelectorAll(selector)
    texts = []

    for element in text_elements:
        content = await page.evaluate('(element) => element.textContent', element)
        texts.append(content.strip())

    await browser.close()

    with open(f"twitch-{channel_name}.json", "w") as f:
        json.dump(texts, f)

    return texts
async def twitch_comments(streamers, sentenceComponentFormData):
    sentence = sentenceComponentFormData['setences'][0]
    pattern = r"\[([^\]]+)\]"
    match = re.search(pattern, sentence)
    streamers =  match[0][1:-1].replace('\'', '').split(',')
    streamers = [removeWhiteSpace(s) for s in streamers] 
    # for streamer in streamers:
    #     subprocess.run(['node', 'RPC/fetch-twitch.js', streamer])
    # return [json.load(open(f'twitch-{streamer}.json', 'r')) for streamer in streamers]     
    # loop = asyncio.get_event_loop()
    # loop.create_task(main())
    #create 3 threads and every 3 seconds poll chat for new messages
    #diff the messages and timestamp new ones 
    tasks = [get_html(streamer) for streamer in streamers]
    results = await asyncio.gather(*tasks)
    return results

def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name
app.mount("/demo", StaticFiles(directory="vite-project/dist/assets"), name="demo")

app.mount("/data/airbnb/h3_poi/", StaticFiles(directory="data/airbnb/h3_poi/"), name="demo")

def assignPeopleToAirbnbBasedOnPreferences():
     makePref = lambda _: [random.random(), random.random(), random.random()]
     [makePref() for i in range(50)]
     # library
     # shopping
     # bar
     #find the count within the geojson and 
     #20 people, 5 airbnbs
   #airbnb ->
        #library : 0 or 1
        # shopping = count within / max of 10
        # bar = count within / max of 10
  
        # one airbnb has 50 bars,
        # total is 100 
         #hayes, dogpatch, sunset,  bernal heights, presidio,
         #shinjuku, fuji, hokaido, narita, nagasaki
        # .5, .1, .25, .15,  0 bars
        #  0, 1, 1, 1, 1, library
        #  3, .3, .3, .3, 0 shopping

        #person_one = hayes, shinjuku 
        #person_two = presidio, nagasaki
        #strong preferences allocated first, then weak preferences allocated second
        # while all not allocated
        # for each person:
        #     for idx each apt:
        #         if closeEnough(person, apt): 
        #            apt.remove(idx)
        # closeEnoughThreshold -= .01
        #             if airbnb[0] - person[0] < .1 allocateshere
        # airbnbs = [shinjuku, fuji, hokaido, narita, nagasaki]
        # shinjuku = [.5, 0, 3]
        #given 12 people
        #12 cities
        # choose the optimal airbnb for each thats within walking distance of each other
        # person_one = [0, .5, .5]
        # person_two = [1, .5, 0]
        # person_three = [.5, .5, .5]
     # coffee w/ lan
     # land usage - green vs urban or proximity to park
     #optimal commute between all friends (5 addresses?)
     #150 million people who live in cities who may want data driven decision making to choose a better appartment that would save them time commuting 
     return makePref()

class UserInDB(BaseModel):
    _k: str
    _v: str

class FnText(BaseModel):
    fn: list[str]
    documentContext: dict
    hashUrl: Optional[str] = ''

#document write on client -> sentenceComponent
#server says -> data[0:5] + componentType + documentContext 4 state
#ui state writes to documentContext -> rerun functions that read from it
#documentContext.suitabilitySliders = {coffee: 1, library: 1}


@app.post("/makeFn")
async def makeFn(FnText:FnText):
    #print('FnText', FnText)
    sentences = FnText.fn
    if len(FnText.hashUrl):
        sentences = json.load(open('documents/' + FnText.hashUrl))
    functions = [substitute(fn) for fn in sentences]
    
    val = False
    args = []
    documentContext = FnText.documentContext
    print('documentContext', documentContext)
    for i, fn in enumerate(functions): 
        if type(fn) == type(lambda _:_):
            #print(fn.__name__)
            if inspect.iscoroutinefunction(fn):
                val = await fn(val, documentContext, sentences[i])
            else:
                val = fn(val, documentContext, sentences[i])
        else:
            val = fn 
        args.append(val)
    return {'fn': args, 'documentContext': documentContext, 'isFromDisk': len(FnText.hashUrl) > 0 }

@app.post("/callFn")
async def admin(request: Request):
    #print('val', await request.json())
    json_data = await request.json()
    city_name = 'Tokyo--Japan'
    def rankApt(personCoefficentPreferences, apt):
        diff = 0
        for key in personCoefficentPreferences:
            if key not in apt: continue
            diff += abs(apt[key] - personCoefficentPreferences[key])
        #print(diff)
        return diff 
    cityAptChoice = {
        'url':'https://www.airbnb.com/rooms/33676580?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-10-25&check_out=2023-10-30&source_impression_id=p3_1695411915_xw1FKQQa0V7znLzQ&previous_page_section_name=1000&federated_search_id=fec99c3c-b5f1-4547-9dda-2bc7758aec94'
    }
    personCoefficentPreferences = json_data['getCoefficents']

    apt_list = json.load(open(f'data/airbnb/apt/{city_name}.json'))[:50]

    def get_json_if_possible(apt):
        if os.path.exists(f'data/airbnb/geocoordinates/{get_room_id(apt)}_geoCoordinates.json'):
            data = json.load(open(f'data/airbnb/geocoordinates/{get_room_id(apt)}_geoCoordinates.json'))
            if (len(data) > 0): 
                data = data[0]
                data = data.split(':')
                data[0] = float(data[0])
                data[1] = float(data[1])
                return data
            else: return [0,0]
        else:
            return [0, 0]

    geocoordinates = [get_json_if_possible(apt) for apt in apt_list]

    coefficents = {'coffee': 1, 'library': 0, 'bar': .5}
    keys = coefficents.keys()

    apts  = []

    import random
    for idx, _ in enumerate(geocoordinates): 
        #print(idx)
        apt = {
            'url': apt_list[idx],
            'loc': geocoordinates[idx]
        } 
        for key in keys:
            coords = _
            apt[key] = random.random()
            #len(fetch_coffee_shops(coords[0], coords[1], [key]))
        apts.append(apt)

    from collections import defaultdict
    totals = defaultdict(int)
    for apt in apts: 
        for key in keys: 
            totals[key] += apt[key]

    for apt in apts: 
        for key in keys: 
            if totals[key] == 0: totals[key] += .01
            apt[key] = apt[key] / totals[key]
    return sorted(apts, key=lambda apt: rankApt(personCoefficentPreferences, apt))[0]

def makePercents(l):
    max_ = max(l)
    return [_ / max_ for _ in l]

@app.get("/admin")
async def admin():
    cells.clear()
    return FileResponse('admin.html')

@app.get("/")
async def metrics():
    return FileResponse('index.html')

@app.get("client.js")
async def js():
    return FileResponse('client.js', media_type="application/javascript")

@app.get("/data/george.txt")
async def george():
    return FileResponse('/data/george.txt', media_type="text/plain")



import uuid 

@app.post("/share")
async def makeFn(fnText:FnText):
    id = str(uuid.uuid4())
    print(fnText, id)
    with open('documents/' + id, 'w+') as f:
        json.dump(fnText.fn, f)
    return id

