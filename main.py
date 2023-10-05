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
def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name
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

#app.mount("/data/airbnb/h3_poi/", StaticFiles(directory="data/airbnb/h3_poi/"), name="demo")
class FnText(BaseModel):
    fn: list[str]
    documentContext: dict
    hashUrl: Optional[str] = ''


print('sentences')
@app.post("/makeFn")
async def makeFn(FnText:FnText):
    sentences = FnText.fn
    print(sentences)
    documentContext = {}
    # args = [
    #     jupyter_functions['given a favorite pokemon']('', '', '')
    # ]
    #return {'fn': args, 'documentContext': documentContext, 'isFromDisk': len(FnText.hashUrl) > 0 }
    if len(FnText.hashUrl):
        sentences = json.load(open('documents/' + FnText.hashUrl))
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
def admin(): return FileResponse('admin.html')
print('sentences')
