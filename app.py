"""
Application definition
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fastapi.middleware.cors import CORSMiddleware
from compiled_functions import jupyter_functions
import inspect
from fastapi import Request, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Optional
import json
import openai
from pyngrok import ngrok
import subprocess
# import fastapi_vite

# templates = Jinja2Templates(directory='templates')
# templates.env.globals['vite_hmr_client'] = fastapi_vite.vite_hmr_client
# templates.env.globals['vite_asset'] = fastapi_vite.vite_asset


#Set configuration options
config = {
    'name': 'example',
    'proto': 'http',
    'addr': 8000,
    'host_header': 'rewrite',
    'subdomain': 'pypypy'
}
import os


# # # Open a HTTP tunnel on the specified port

if os.getcwd() != '/Users/shelbernstein/cooperation.party':
    for i in range(10): print('this is fly.io')
    ngrok.set_auth_token('2TUCQ8cPQuaI0FJDPRhOXrxeEl3_81nTfvqtKfv9TYpCvAzBE')
    public_url = ngrok.connect(**config)
else: 
    for i in range(10): print('this is mbp')
    ngrok.set_auth_token('2TUCQ8cPQuaI0FJDPRhOXrxeEl3_81nTfvqtKfv9TYpCvAzBE')
    config['subdomain'] = 'shelbernstein'
    public_url = ngrok.connect(**config)


# import fastapi_vite

# templates = Jinja2Templates(directory='templates')
# templates.env.globals['vite_hmr_client'] = fastapi_vite.vite_hmr_client
# templates.env.globals['vite_asset'] = fastapi_vite.vite_asset

#public_url = ngrok.connect(**config)
# #print(f"ngrok tunnel '{config['name']}' is running at {public_url}")
# # Keep the tunnel running until terminated
# #input("Press Enter to exit...")
# # Terminate the ngrok tunnel when the script is interrupted
# #ngrok.kill()
def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name

origins = [
    "https://cooperation.party",
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
    "https://localhost:5173"  
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount("/data/airbnb/h3_poi/", StaticFiles(directory="data/airbnb/h3_poi/"), name="demo")
class FnText(BaseModel):
    fn: list[str]
    documentContext: dict
    hashUrl: Optional[str] = ''


class neighborhoodDetails(BaseModel):
    apt_url: str
    schedule_text: str
    city_name: str


def schedule_json_converter(_):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": """convert unstructured data to json with this schema
        {
        "vending_machine": {
            "days_per_week": 3,
        },
        "post_office": {
            "days_per_week": 4
        },
        "cafe": {
            "days_per_week": 2
        }    
        }
        if it says vending_machine 3x per week
        """
        },
        {
        "role": "user",
        "content": _
        },
        {
        "role": "assistant",
        "content": ""
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    result = json.loads(response['choices'][0]['message']['content'])
    print(_, 'scheduling result', result)
    return result

from getRoutes import getRoutes 
@app.post("/neighborhoodDetails")
async def callFn(neighborhoodDetails: neighborhoodDetails):
    print('this is my neighborhoodDetails', neighborhoodDetails.schedule_text)
    schedule = schedule_json_converter(neighborhoodDetails.schedule_text)
    routes, time =  getRoutes(neighborhoodDetails.city_name, neighborhoodDetails.apt_url, schedule)
    return {'key': routes, 'schedule_places': schedule}
    return {'key': 123}

@app.post("/makeFn")
async def makeFn(FnText:FnText):
    sentences = FnText.fn
    documentContext = {}
    # args = [
    #     jupyter_functions['given a favorite pokemon']('', '', '')
    # ]
    #return {'fn': args, 'documentContext': documentContext, 'isFromDisk': len(FnText.hashUrl) > 0 }
    # if len(FnText.hashUrl):
    #     sentences = json.load(open('documents/' + FnText.hashUrl))
    functions = [substitute(fn) for fn in sentences]
    
    val = False
    args = []
    documentContext = FnText.documentContext
    print('documentContext', documentContext)
    for i, fn in enumerate(functions): 
        if type(fn) == type(lambda _:_):
            if inspect.iscoroutinefunction(fn):
                val = await fn(val, documentContext, sentences[i])
            else:   
                val = fn(val, documentContext, sentences[i])
        else:
            val = fn 
        args.append(val)
    return {'fn': args, 'documentContext': documentContext, 'isFromDisk': len(FnText.hashUrl) > 0 }

@app.get("/admin")
def admin(): return FileResponse('./templates/admin.html')

@app.get("/ls")
async def home():
    called = subprocess.run(["ls", "-l"], capture_output=True)
    return HTMLResponse("Hello happy healthy and safe world!" + str(called.stdout))


def get_airbnb_data_for_city(city_name: str):
    #return [{"id": 1, "name": "Sample Airbnb 1"}, {"id": 2, "name": "Sample Airbnb 2"}]
    with open(f'data/airbnb/apt/{city_name}.json') as file:
        data = list(json.load(file).items())[:1000]
    return data

@app.get("/data/airbnb/apt/{city_name}")
async def get_airbnb_for_city(city_name: str):
    # Fetch data using the helper function
    data = get_airbnb_data_for_city(city_name)
    print(city_name)
    # Check if data is present, if not, return an error
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city_name}")
    
    return data

import os
@app.get("/cityList")
async def get_list_of_cities_with_airbnbs_hopefully():
    # Fetch data using the helper function
    data = [_.replace('.json', '') for _ in  os.listdir('data/airbnb/apt')]
    # Check if data is present, if not, return an error
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city_name}")

    return data

from fastapi.responses import StreamingResponse
import time
async def stream_content():
    headers = {
        "X-Total-Chunks": str(3173957)  # This is another way to signal the total chunks
    }
    # with open("data/city_locations.json", "r") as f:
    #     yield f"{f.readline()}\n"
    count = 0
    with open("data/worldcities.csv", "r") as f:
        for line in f:
            count += 1
            print('wtf')
            if count > 1:
                yield line
        #yield f"{f.readline()} i like poo\n"

import requests 
def fetch_coworking(min_lat, min_lng, max_lat, max_lng):
    places = []
    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="bench"]({min_lat},{min_lng},{min_lat + 1},{min_lng + 1});
    );
    out body;
    """ 

    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="bench"]({34},{138},{35},{139});
    );
    out body;
    """ 
    import random
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})
    #print(response.status_code, longitude, latitude, amenities)
    if response.status_code == 200:
        data = response.json()
        coffee_shops = data['elements']
        places += coffee_shops
    else: 
        print('we are shit out of luck', response.status_code)
        return []
    random.shuffle(places)
    return places[:100]


import aiohttp
import asyncio
async def getAllRoutesFaster():
    routes = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, apt) for city in apt_json for apt in apt_json[city]  ]
        print(len(tasks))
        routes = await asyncio.gather(*tasks)
        #apt_json[city] = sorted(apt_json[city], key=lambda _: _['good_deal'])[:100]
    #apt_json, h3_complaints = compute_311(apt_json)
    #print(h3_complaints)
    #apt_json, routes = compute_travel_time(apt_json, schedule)
    return routes

async def fetch(session, apt):
    routes = []
    travel_time = 0
    start_lng = float(apt['longitude'])
    start_lat = float(apt['latitude'])
    for todo in schedule:
        result = await fetch_overpass_data(todo, start_lat, start_lng)
        await asyncio.sleep(1.5)
        if not result or 'elements' not in result or len(result['elements']) == 0: 
            print('no ' + todo)
            continue
        result = result['elements'][0]
        print('result', result)
        end_lng = result['lon']
        end_lat = result['lat']
        url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{start_lng}%2C{start_lat}%3B{end_lng}%2C{end_lat}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
        async with session.get(url) as response:
            route = await response.json()
            if route and len(route['routes']) > 0: 
                travel_time += route['routes'][0]['duration']
                routes.append(route)
    apt['commute_distance'] = travel_time
    return routes

def fetchRoad(start, end):
    print(start, end)
    start_lng = start['lon']
    start_lat = start['lat']
    end_lng = end['lon']
    end_lat = end['lat']
    url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{start_lng}%2C{start_lat}%3B{end_lng}%2C{end_lat}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else: 
        print('we are shit out of luck', response.status_code)
        return []

#use query params
prev = time.time()
@app.get("/osm_bbox/")
async def stream(min_lat:float, min_lng:float, max_lat:float, max_lng:float):
    global prev
    if abs(time.time() - prev) < 1:
        prev = time.time()
        print('too soon')
        return {'places': [], 'routes': []}
    prev = time.time()
    places = fetch_coworking(min_lat, min_lng, max_lat, max_lng)
    routes = []
    for place in places[:5]: 
        for place_two in places[5:10]:
            routes.append(
                fetchRoad(
                    place, place_two
                )
            )
    print(len(places))
    return {
        'places': places,
        'routes': routes
    }
    #return StreamingResponse(stream_content(), media_type="text/plain")

# @app.get("/data/airbnb/apt/")
# async def home():
#     called = subprocess.run(["ls", "-l"], capture_output=True)
#     return HTMLResponse("Hello happy healthy and safe world!" + str(called.stdout))

# @app.get("/")
# async def index():
#     StaticFiles(
#         '*',
#         directory=None,
#         packages=None,
#         html=False,
#         check_dir=True,
#         follow_symlink=False



from fastapi.staticfiles import StaticFiles


# StaticFiles(
#     '*',
#     directory=None,
#     packages=None,
#     html=False,
#     check_dir=True,
#     follow_symlink=False
# )

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

# def SPA(app: FastAPI, build_dir: Union[Path, str]) -> FastAPI:
# # Serves a React application in the root directory

#     @app.exception_handler(StarletteHTTPException)
#     async def _spa_server(req: Request, exc: StarletteHTTPException):
#         if exc.status_code == 404:
#             return FileResponse(f'{build_dir}/index.html', media_type='text/html')
#         else:
#             return await http_exception_handler(req, exc)

#     if isinstance(build_dir, str):
#         build_dir = Path(build_dir)

#     app.mount(
#         '/vite-app/dit',
#         StaticFiles(directory=build_dir / 'dist'),
#         name='React app static files',
#     )