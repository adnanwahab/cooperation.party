from flask import Flask, render_template,send_from_directory
from flask_cors import CORS 
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from compiled_functions import jupyter_functions
import inspect
import json

app = Flask(__name__, 
             static_folder="last-day", )

#app = Flask(__name__)
CORS(app, resources={r"*": {"origins": [
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
]}})

@app.route('/')
def hello(name=None):
    print('hello world')
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/makeFn', methods=['POST'])
def makeFn():
    print('MAKE FN')
    data = request.json
    sentences = data.get("fn", [])
    if data.get("hashUrl"):
        with open('documents/' + data["hashUrl"]) as f:
            sentences = json.load(f)
    
    functions = [substitute(fn) for fn in sentences]
    
    val = False
    args = []
    print(functions)
    documentContext = data.get("documentContext", {})
    for i, fn in enumerate(functions): 
        if type(fn) == type(lambda _:_):
            if inspect.iscoroutinefunction(fn):
                print('is not here')
                val = fn(val, documentContext, sentences[i])  # Note: Flask does not natively support async
            else:
                val = fn(val, documentContext, sentences[i])
        else:
            val = fn 
        args.append(val)
    print('args', args)
    return jsonify({
        'fn': args,
        'documentContext': documentContext,
        'isFromDisk': bool(data.get("hashUrl"))
    })


@app.route('/assets/<path>')
def serve_static(path):
    return send_from_directory(app.static_folder + '/assets/', path)

@app.route('/wtf')
def no(name=None):
    print('hello world')
    return "<p>wtf</p>"
def substitute(name):
    for k in jupyter_functions:
        if (len(name)) < 1: return name
        if name[0] == '#': return name
        if k in name:
            return jupyter_functions[k]
    return name

# @app.route('/admin')
# def admin():
#     return send_from_directory('', 'admin.html')

# if __name__ == '__main__':
#     app.run(debug=True)

print('cooperation.party')