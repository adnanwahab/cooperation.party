from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exception_handlers import http_exception_handler
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
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
from getRoutes import getRoutes 
config = {
    'name': 'example',
    'proto': 'http',
    'addr': 8000,
    'host_header': 'rewrite',
    'subdomain': 'pypypy'
}

if os.getcwd() != '/Users/shelbernstein/cooperation.party':
    for i in range(10): print('this is fly.io')
    ngrok.set_auth_token('2TUCQ8cPQuaI0FJDPRhOXrxeEl3_81nTfvqtKfv9TYpCvAzBE')
    #public_url = ngrok.connect(**config)
else: 
    for i in range(10): print('this is mbp')
    ngrok.set_auth_token('2TUCQ8cPQuaI0FJDPRhOXrxeEl3_81nTfvqtKfv9TYpCvAzBE')
    config['subdomain'] = 'shelbernstein'
    public_url = ngrok.connect(**config)

def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name

origins = [
    "https://groundbear.static.observableusercontent.com",
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
class FnText(BaseModel):
    fn: list[str]
    documentContext: dict
    hashUrl: Optional[str] = ''

class neighborhoodDetails(BaseModel):
    apt_url: str
    schedule_text: str
    city_name: str

from nbformat import v4 as nbf
from nbclient import NotebookClient


from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/notebooks")
def read_notebooks():
    return RedirectResponse(url="http://127.0.0.1:8888/notebooks")

@app.get("/execute")
async def execute_code():
    code = 'print(500)'
    # Create a new notebook object
    nb = nbf.new_notebook()
    nb.cells.append(nbf.new_code_cell(code))

    # Set up the client to execute the notebook
    # It will use the kernel specified in your current Jupyter environment
    client = NotebookClient(nb, timeout=60, kernel_name='python3')

    try:
        # Execute the notebook
        client.execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract the output from the last cell (assuming it's the code cell)
    output = client.nb.cells[-1].outputs

    # Process the output as needed, here we're just returning it as a string
    return {"output": output}

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

@app.post("/neighborhoodDetails")
async def callFn(neighborhoodDetails: neighborhoodDetails):
    print('this is my neighborhoodDetails', neighborhoodDetails.schedule_text)
    schedule = schedule_json_converter(neighborhoodDetails.schedule_text)
    routes, time =  getRoutes(neighborhoodDetails.city_name, neighborhoodDetails.apt_url, schedule)
    return {'key': routes, 'schedule_places': schedule}

@app.post("/makeFn")
async def makeFn(FnText:FnText):
    sentences = FnText.fn
    documentContext = {}
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


@app.get("/current_frame")
def admin(): return FileResponse('./frame.jpg')

@app.get("/ipad")
def admin(): return FileResponse('./templates/ipad-face.html')

from fastapi import FastAPI, File, UploadFile
from typing import Annotated


@app.post("/cameraframe")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    with open(f"{file.filename}", "wb") as f:
        f.write(content)
    return JSONResponse(content={"message": "File received"})
    #if file.content_type == "image/jpeg" or file.content_type == "image/png":
        #file_path = os.path.join(os.path.dirname(__file__), "static", file.filename)
        #print(file.content_type)
    # with open('camera.jpg', 'wb+') as buffer:
    #     buffer.write(file.read())
    # return {'message': 'thanks'}
    #     return {"message": f"File {file.filename} uploaded"}
    # else:
    #     return {"error": "Invalid file type"}

# @app.post("/cameraframe")
# def cameraframe(data): 
#     #return FileResponse('./templates/ipad-face.html')
#     print('data', data)
#     with open('camera.png', 'w+') as file:
#         file.write(data)
#     return {'thankyou': "denada"}


@app.get("/admin")
def admin(): return FileResponse('./templates/admin.html')

@app.get("/ls")
async def home():
    called = subprocess.run(["ls"], capture_output=True)
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
#find out max requests sent to both overpass + mapbox

def fetch_coworking(min_lat, min_lng, max_lat, max_lng):
    places = []
    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="bench"]({max_lat},{max_lng},{min_lat},{min_lng});
    );
    out body;
    """ 
    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="bench"]({min_lat},{min_lng},{max_lat},{max_lng});
    );
    out 100;
    """ 
    #print(query)

    # query = f"""
    # [out:json][timeout:25];
    # (
    #     node["amenity"="bench"]({min_lng},{min_lat},{max_lng},{max_lat});
    # );
    # out body;
    # """ 
    #     (._;._;);
    # out 0..100;

    # query = f"""
    # [out:json][timeout:25];
    # (
    #     node["amenity"="bench"]({34},{138},{35},{139});
    # );
    # out body;
    # """ 
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
        print(response.text)
        return []
    random.shuffle(places)
    return places
    #return places[:100]

import aiohttp
import asyncio

async def getAllRoutesFaster(places):
    routes = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, place_1, place_2) for place_1 in places[:5] for place_2 in places[10:15]  ]
        routes = await asyncio.gather(*tasks)
    return routes

async def fetch(session, place_1, place_2):
    routes = []
    start_lng = place_1['lon']
    start_lat = place_1['lat']
    end_lng = place_2['lon']
    end_lat = place_2['lat']
    url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{start_lng}%2C{start_lat}%3B{end_lng}%2C{end_lat}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    async with session.get(url) as response:
        route = await response.json()
        if route and len(route['routes']) > 0: 
            routes.append(route)
    return routes

prev = time.time()
@app.get("/osm_bbox/")
async def stream(min_lat:float, min_lng:float, max_lat:float, max_lng:float):
    global prev
    if abs(time.time() - prev) < 1:
        prev = time.time()
        return {'places': [], 'routes': []}
    prev = time.time()
    places = fetch_coworking(min_lat, min_lng, max_lat, max_lng)
    routes = []
    routes = await getAllRoutesFaster(places)
    return {
        'places': places,
        'routes': routes
    }

app.mount("/", StaticFiles(directory="vite-project/dist", html=True), name="frontend")

@app.get("/get_apt_details/")
async def get_apt_details(apt_id: str):
    print('get_apt_details', apt_id)
    listing_to_city_name = json.load(open('data/listing_to_city_name.json'))
    city_name = listing_to_city_name[apt_id]
    #given an apt id, find the city_name
    #city_apt_details = json.load(open(f'data/airbnb/columns/{city_name}'))
    city_apt_details = json.load(open(f'/Users/shelbernstein/cooperation.party/data/airbnb/columns/Columbus--Ohio--United-States.json'))
    print(city_apt_details)
    return city_apt_details['90676']
    return city_apt_details[apt_id]