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
import json
import os.path
import nltk
from nltk.corpus import words
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('words')

def key_function (_):
    return _[1]


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

isochroneLibraryCache = {}

def isochroneLibrary(longitude, latitude, listing):
    if latitude in isochroneLibrary:  
        return isochroneLibraryCache[latitude]
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
        isochroneLibraryCache[latitude] = [data, geojson_data, latitude, longitude] 
        return [data, geojson_data, latitude, longitude]
    else : return False

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

def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

def map_of_all_airbnbs(_,__, i):
    cities = [json.load(open(_)) for _ in  glob.glob('data/airbnb/apt/*')]
    geoCode = [json.load(open(f'data/airbnb/geocoordinates/{get_room_id(listing)}.json')) 
               for city in cities 
               for listing in city 
               if os.path.exists(f'data/airbnb/geocoordinates/{get_room_id(listing)}')
               ]
    return {'data': geoCode, 'component': '<map>', 
            'geoCoordCache': geoCoordCache
            
            }

def filter_by_poi(_, sentenceComponentFormData, i):
    print('sentenceComponentFormData', sentenceComponentFormData)
    poi = 'coffee'
    poi = sentenceComponentFormData['sentences'][i].strip().split(' ')[2]
    print('poi', poi)
    #fetch_coffee_shops() -> return a curry with fetch_coffee_shops that finds cached POI percentage? for the previous ones

    #function should return _ on server side compose
    #on client returns a slider which edits the componentFormData -> which runs a filter on the server 
    if (_ == 'hello-world'): return {'component': '<slider>', 'data': _, 'label': poi}
    if (type(_) is not list): _ = _['data']
    return {'component': '<slider>', 'data': _, 'label': poi}


def groupBySimilarity(sentences, documents, i):
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

def continentRadio(_, __, i):
    from ipynb.fs.defs.geospatial import getCityList
    return {
        'key':'continent',
        'data': ['Europe', 'North America', 'Asia', 'South America', 'Africa', 'Australia and Oceania', 'Others/Islands'], 
        'component': '<Radio>'
        }

def cityRadio(_, __, i):
    from ipynb.fs.defs.geospatial import getCityList
    return {'key':'city','data': getCityList(), 'component': '<Radio>'}

def getAirbnbs(_, componentData, i):
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

def filter_by_distance_to_shopping_store(airbnbs, documentContext, i):
    if (type(airbnbs) is dict): airbnbs = airbnbs['data']
    #print('airbnbs!', airbnbs)
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
    gm_urls = [json.load(open(fp)) for fp in [f"data/airbnb/gm/{get_room_id(apt_url)}.json" for apt_url in airbnbs
                                              ]
                                              if os.path.exists(fp)
                                              ]
    geoCoordinates = [imageToCoords(_, documentContext['city'], get_room_id(airbnbs[idx]) ) for idx, _ in enumerate(gm_urls[:18])]

    geoCoordinates = [coord[0].split(':') for coord in geoCoordinates if len(coord) > 1]
    _ = [isochroneLibrary(pt[0], pt[1], get_room_id(airbnbs[idx])) for idx, pt in enumerate(geoCoordinates)]

    return [_ for _ in _ if _ != False]  

def getYoutube(url, i):
    youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'}).download(['https://www.youtube.com/watch?v=a02S4yHHKEw&ab_channel=AllThatOfficial'])
    audio_file= open("audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    open('transcript.txt', 'w').write(transcript)
    return '<audio src="audio.mp3">'

def poll(_, second, i):
    return 'lots of cool polling data'
    return open('./poll.json', 'r').read()

def arxiv (_, sentence, i):
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

def trees_histogram(_, sentence, i):
    from ipynb.fs.defs.geospatial import trees_histogram
    print('trees histogram')
    return trees_histogram()

def twitch_comments(_, sentence, i):
    return json.load(open('./data/twitch.json', 'r'))


def getTopics(sentences, sentence, i):
    counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence.split(' '):
            counts[word] += 1
    topics = []
    for k in counts:
        if counts[k] > 2:
            topics.append(k)
    return topics

def trees_map(_, sentence, i):
    from ipynb.fs.defs.geospatial import trees_map
    return trees_map()[:100000]

def pokemon(_, __, i):
    from ipynb.fs.defs.Pokemon_Dota_Arxiv import generate_team
    return generate_team()

def satellite_housing(_, sentence):
    requests.get('https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/static/-122.4241,37.78,14.25,0,60/600x600?access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg')
    return 'for each satellite images in area find anything that matches criteria'




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


jupyter_functions = { 
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
    'map of all airbnbs': map_of_all_airbnbs,
    'i like ': filter_by_poi
}

