# Paris, France: [48.8566, 2.3522]
# Rome, Italy: [41.9028, 12.4964]
# Barcelona, Spain: [41.3851, 2.1734]
# Amsterdam, Netherlands: [52.3676, 4.9041]
# London, United Kingdom: [51.5074, -0.1278]
# Prague, Czech Republic: [50.0755, 14.4378]
# Vienna, Austria: [48.2082, 16.3738]
# Budapest, Hungary: [47.4979, 19.0402]
# Berlin, Germany: [52.5200, 13.4050]
# Athens, Greece: [37.9838, 23.7275]
# Venice, Italy: [45.4408, 12.3155]
# Lisbon, Portugal: [38.7223, -9.1393]
# Copenhagen, Denmark: [55.6761, 12.5683]
# Stockholm, Sweden: [59.3293, 18.0686]
# Edinburgh, Scotland: [55.9533, -3.1883]
# Dublin, Ireland: [53.3498, -6.2603]
# Reykjavik, Iceland: [64.1466, -21.9426]
# Madrid, Spain: [40.4168, -3.7038]
# Oslo, Norway: [59.9139, 10.7522]
# Zurich, Switzerland: [47.3769, 8.5417]
# North America:

# New York City, USA: [40.7128, -74.0060]
# San Francisco, USA: [37.7749, -122.4194]
# Vancouver, Canada: [49.2827, -123.1207]
# New Orleans, USA: [29.9511, -90.0715]

from shapely.geometry import shape, Point
import random 
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import requests
import easyocr
from fastapi import Request, FastAPI
import random
import json 
import subprocess
import json
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import youtube_dl
import openai
import re
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
from pydantic import BaseModel

import json
import os.path
import nltk
from nltk.corpus import words
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('words')
def removeWhiteSpace(_):
    return _.strip()

def getYoutube(url):
    youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'}).download(['https://www.youtube.com/watch?v=a02S4yHHKEw&ab_channel=AllThatOfficial'])
    audio_file= open("audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    open('transcript.txt', 'w').write(transcript)
    return '<audio src="audio.mp3">'

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
    print('fnText', fnText)
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

def key_function (_):
    return _[1]

def imageToCoords(url_list, location='_', apt_url='_'):
    fp = f'data/airbnb/geocoordinates/{apt_url}_geoCoordinates.json'
    print('reading cache ', os.path.exists(fp))
    if os.path.exists(fp):
        return json.load(open(fp, 'r'))
    cache = set()
    for _ in url_list[:5]:
        response = requests.get(_)
        if response.status_code == 200:
            with open(_[-50:-1], 'wb') as f:
                f.write(response.content)
        print('OCR', fp)
        ocr = ocrImage(_[-50:-1])
        if not ocr: continue
        coords = geoCode(ocr, location)
        if not coords: continue
        cache.add(str(coords[0]) + ':' + str(coords[1]))
    print ('writing to' + fp)
    json.dump(list(cache), open(fp, 'w'))
    return list(cache)

def ocrImage(fp):
    reader = easyocr.Reader(['en'])
    extract_info = reader.readtext(fp)
    from time import time
    sorted(extract_info, key=key_function)
    if (not extract_info): return False
    return extract_info[0][1]   

def geoCode(address, city):
    accessToken = "pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"  # Replace with your actual access token
    geocodeUrl = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}%2C%20{city}.json?access_token={accessToken}"
    response = requests.get(geocodeUrl)
    data = response.json()
    if 'features' in data and len(data['features']) > 0:
        location = data['features'][0]['geometry']['coordinates']
        return location

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
    print('processMessages', result)
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
    print(message.text)
    return {'demo': exec(message.text)}

@app.get("/nexus")
async def index():
    return JSONResponse(content={'456':123})

class FnText(BaseModel):
    fn: list[str]
    sentenceComponentData: dict

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
def poll(_, second):
    return 'lots of cool polling data'
    return open('./poll.json', 'r').read()

hasRendered = False
hasRenderedContent = False
def arxiv (_, sentence):
    global hasRendered, hasRenderedContent  # Declare as global to modify
    print('ARXIV')
    import pdfplumber
    import glob
    fileList = glob.glob('./*.pdf')[:3]
    print(fileList)
    content = []
    if hasRendered: return hasRenderedContent
    for f in fileList:
        print(f)
        with pdfplumber.open(f) as pdf:
            # Loop through each page
            for i, page in enumerate(pdf.pages):
            # Extract text from the page
                text = page.extract_text()
                content.append(text)
            #print(f'Content from page {i + 1}:\n{text}')
    hasRendered = True
    hasRenderedContent = content
    return content

def trees_map(_, sentence):
    from ipynb.fs.defs.geospatial import trees_map
    return trees_map()[:100000]

def trees_histogram(_, sentence):
    from ipynb.fs.defs.geospatial import trees_histogram
    print('trees histogram')
    return trees_histogram()

def twitch_comments(_, sentence):
    return json.load(open('./data/twitch.json', 'r'))

def getTopics(sentences, sentence):
    counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence.split(' '):
            counts[word] += 1
    topics = []
    for k in counts:
        if counts[k] > 2:
            topics.append(k)
    return topics

def continentRadio(_, __):
    from ipynb.fs.defs.geospatial import getCityList
    return {
        'key':'continent',
        'data': ['Europe', 'North America', 'Asia', 'South America', 'Africa', 'Australia and Oceania', 'Others/Islands'], 
        'component': '<Radio>'
        }

def cityRadio(_, __):
    from ipynb.fs.defs.geospatial import getCityList
    return {'key':'city','data': getCityList(), 'component': '<Radio>'}


geoCoordCache = {}
def fetch_coffee_shops(longitude, latitude, amenities = ['cafe', 'library', 'bar']):
    if round(longitude, 1) in geoCoordCache: 
        print('WE GOT THE CACHE', len(geoCoordCache[round(longitude, 1)]))
        return geoCoordCache[round(longitude, 1)]
    # if (os.path.exists(f'data/airbnb/poi/{longitude}_{latitude}_places.json')):
    #     return json.load(open(f'data/airbnb/poi/{longitude}_{latitude}_places.json', 'r'))
   
    places = []
    for i in amenities:
        query = f"""
        [out:json][timeout:25];
        (
            node["amenity"="{i}"]({latitude - 0.01},{longitude - 0.01},{latitude + 0.01},{longitude + 0.01});
        );
        out body;
        """ 
        overpass_url = "https://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={'data': query})
    
        if response.status_code == 200:
            data = response.json()
            coffee_shops = data['elements']
            places += coffee_shops
    if len(places) > 0:
        #print(places)
        geoCoordCache[round(longitude, 1)] = places
    #json.dump(places, open(f'data/airbnb/poi/{listing}_places.json', 'w'))
    return places

def getAirbnbs(_, componentData='cairo, egypt'):
    from ipynb.fs.defs.geospatial import getAllApt_
    if 'city' not in componentData: return 'hello-world'
    location = componentData['city']
    location = location.replace(', ', '--')
    if location == '': return 'hello-world'
    fp = f'data/airbnb/apt/{location}.json'
    fp_gm = lambda apt: f'/data/airbnb/gm/{get_room_id(apt)}.json'
    # if os.path.exists(fp):
    #     apts = json.load(open(fp, 'r'))
    #     gm = [os.path.exists(fp_gm(apt)) for apt in apts]
    #     gm = [_ for _ in gm if _ == True]
    #     print(len(apts) == len(gm), 'all apt found')
    #     if len(apts) == len(gm) and len(gm) != 0: return [apt['link'] for apt in apts]
    args = [
        "node",
        "rpc/getAptInCity.js",
        location
    ]
    #completed_process = subprocess.run(args)
    args = [
        "node",
        "rpc/airbnb_get_img_url.js",
        f'data/airbnb/apt/{location}.json'
    ]
    #completed_process = subprocess.run(args)
    #print(location)
    apts = json.load(open(fp, 'r'))

    return [apt for apt in apts]

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
    print('aptGeoLocation', aptGeoLocation)
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
def filter_by_distance_to_shopping_store(airbnbs, documentContext):
    #return ['asdf', 'hello']
    #print(airbnbs, airbnbs)
    #return ['asdf', 'hello']
    #SSreturn airbnbs[:10]
    #subprocess.run(['node', 'airbnb_get_img_url.js', 'jaipur--india_apt.json'])
    if airbnbs =='hello-world': return 'hello world'
    #writes to listing_url.json
    #for each listing_url -> get img url
    #for each img url -> OCR
    #for each OCR -> geocode
    #for each geocode -> get nearby shopping stores
    #sort list of appartments by distance to shopping store
    #make better
    #imageToCoords() #apt_url -> coordinate
    
    #getPlacesOfInterest() #coordiante -> get distance to shopping store
    #print ('airbnbs', airbnbs)
    #for each apt
    #return airbnbs[:10]
    cache = {}
    def doesExist(url):
        if url not in cache: 
            cache[url]  = True
            return True
        return False
    
    airbnbs = [apt for apt in airbnbs if doesExist(apt)]
    print(airbnbs)
    gm_urls = [json.load(open(fp)) for fp in [f"data/airbnb/gm/{get_room_id(apt_url)}.json" for apt_url in airbnbs
                                              ]
                                              if os.path.exists(fp)
                                              ]
    geoCoordinates = [imageToCoords(_, documentContext['city'], get_room_id(airbnbs[idx]) ) for idx, _ in enumerate(gm_urls[:18])]

    geoCoordinates = [coord[0].split(':') for coord in geoCoordinates if len(coord) > 1]
    print(geoCoordinates)
    _ = [isochroneLibrary(pt[0], pt[1], get_room_id(airbnbs[idx])) for idx, pt in enumerate(geoCoordinates)]

    return [_ for _ in _ if _ != False]  
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


def isochroneLibrary(longitude, latitude, listing):
    latitude = float(latitude)
    longitude = float(longitude)  
    print('get all the coffee shops within driving distance of this airbnb') 
    contours_minutes = 15
    contours_minutes = 30
    assert(latitude < 90 and latitude > -90)
    isochrone_url = f'https://api.mapbox.com/isochrone/v1/mapbox/walking/{longitude}%2C{latitude}?contours_minutes={contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    geojson_data = requests.get(isochrone_url).json()

    coffee_shops = fetch_coffee_shops(longitude, latitude)
    data = []
    for shop in coffee_shops: 
        if 'lat' not in shop or 'lon' not in shop: 
            print(shop)
            continue
        point_to_check = Point(shop['lon'], shop['lat'])
        for feature in geojson_data['features']:
            polygon = shape(feature['geometry'])
            if polygon.contains(point_to_check):
                data.append(shop)
    print('cofee shops within geojson', len(data))
    if len(data) > 0: 
        return [data, geojson_data, latitude, longitude]
    else : return False

import requests
def satellite_housing(_, sentence):
    requests.get('https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/static/-122.4241,37.78,14.25,0,60/600x600?access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg')
    return 'for each satellite images in area find anything that matches criteria'

def pokemon(_, __):
    from ipynb.fs.defs.Pokemon_Dota_Arxiv import generate_team
    return generate_team()


def groupBySimilarity(sentences, documents):
    print(sentences)
    from sentence_transformers import SentenceTransformer,util
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = [item for sublist in sentences for item in sublist]
    encodings = model.encode(sentences, convert_to_tensor=True, device='cpu')
    clusters = util.community_detection(encodings, min_community_size=1, threshold=0.55)
    sentenceClusters = []
    for s in clusters:
        sentenceCluster = []
        for id, sentenceId in enumerate(s):
            sentenceCluster.append(sentences[sentenceId])
        sentenceClusters.append(sentenceCluster)
    
    return sentenceClusters

async def delay(time):
    await asyncio.sleep(time)
import pyppeteer
async def get_html(channel_name):
    print(f"hi, {channel_name}")

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

    print(f"texts: {texts}")

    with open(f"twitch-{channel_name}.json", "w") as f:
        json.dump(texts, f)

    return texts
async def twitch_comments(streamers, sentenceComponentFormData):
    print('_', sentenceComponentFormData)
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


import glob
def map_of_all_airbnbs(_,__):
    
    cities = [json.load(open(_)) for _ in  glob.glob('data/airbnb/apt/*')]
    geoCode = [json.load(open(f'data/airbnb/geocoordinates/{get_room_id(listing)}.json')) 
               for city in cities 
               for listing in city 
               if os.path.exists(f'data/airbnb/geocoordinates/{get_room_id(listing)}')
               ]
    return {'data': geoCode, 'component': '<map>'}

jupyter_functions = { #read all functions in directory -> 
    'group them into topics': groupBySimilarity,
    'for each continent': continentRadio,
    'choose a city in each': cityRadio,
    'find all airbnb in that city': getAirbnbs,
    'filter by distance to shopping store': filter_by_distance_to_shopping_store,
    'filter by 10 min train or drive to a library above 4 star': filter_by_distance_to_shopping_store,
    'plot on a map': lambda x,y: x,
    'get transcript from ': getYoutube,
    'poll': poll,
    'plant-trees': lambda _,__: 'put map here',
    'arxiv': arxiv,
    'trees_histogram' : trees_histogram,
    'twitch_comments' : twitch_comments,
    'getTopics': getTopics, 
    'trees_map': trees_map,
    'housing_intersection': 'housing_intersection',

    'for each satellite images in area find anything that matches criteria': satellite_housing,
    'given a favorite pokemon': pokemon,

    'get all twitch comments': twitch_comments,

    'map of all airbnbs': map_of_all_airbnbs
}
import asyncio
import inspect

def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name
app.mount("/demo", StaticFiles(directory="vite-project/dist/assets"), name="demo")

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


@app.post("/makeFn")
async def makeFn(FnText:FnText):
    print('FnText', FnText)
    functions = [substitute(fn) for fn in FnText.fn]
    FnText.sentenceComponentData['sentences'] = FnText.fn
    
    val = False
    args = []
    for i, fn in enumerate(functions): 
        if type(fn) == type(lambda _:_):
            print(fn.__name__)
            if inspect.iscoroutinefunction(fn):
                val = await fn(val, FnText.sentenceComponentData)
            else:
                val = fn(val, FnText.sentenceComponentData)
        else:
            val = fn 
        args.append(val)
    return {'fn': args}


@app.post("/callFn")
async def admin(request: Request):
    print('val', await request.json())
    json_data = await request.json()
    city_name = 'Tokyo--Japan'
    #json['city_name']

    #cities = ['tokyo', 'houston', 'moscow', 'cairo', 'mumbai', 'delhi', 'shanghai', 'beijing', 'dhaka', 'osaka', 'chongqing', 'istanbul']
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

    print('doing some shit')
    import random
    for idx, _ in enumerate(geocoordinates): 
        print(idx)
        apt = {
            'url': apt_list[idx],
            'loc': geocoordinates[idx]
        } 
        for key in keys:
            coords = _
            apt[key] = random.random()
            #len(fetch_coffee_shops(coords[0], coords[1], [key]))
        apts.append(apt)
    print('did some shit')

    from collections import defaultdict
    totals = defaultdict(int)
    for apt in apts: 
        for key in keys: 
            print(totals, apt)
            totals[key] += apt[key]

    for apt in apts: 
        for key in keys: 
            if totals[key] == 0: totals[key] += .01
            apt[key] = apt[key] / totals[key]
    #apt = [[coffee_[idx], bar_[idx], libraries_[idx], _] for idx, _ in enumerate(apt_list)]

    print(apts[:10])
    #return apts[0]
    return sorted(apts, key=lambda apt: rankApt(personCoefficentPreferences, apt))[0]


def makePercents(l):
    max_ = max(l)
    return [_ / max_ for _ in l]

@app.get("/admin")
async def admin():
    cells.clear()
    print('Clearing')
    return FileResponse('admin.html')

@app.get("/")
async def metrics():
    return FileResponse('/vite-project/dist/index.html')

@app.get("client.js")
async def js():
    return FileResponse('client.js', media_type="application/javascript")

@app.get("/data/george.txt")
async def george():
    return FileResponse('/data/george.txt', media_type="text/plain")


