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
        return list(json.load(file).values())[:1000]

@app.get("/data/airbnb/apt/{city_name}")
async def get_airbnb_for_city(city_name: str):
    # Fetch data using the helper function
    data = get_airbnb_data_for_city(city_name)
    print(city_name)
    # Check if data is present, if not, return an error
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for city: {city_name}")
    
    return data


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

from fastapi import FastAPI, Request
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