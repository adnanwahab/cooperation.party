import React, {useRef, useEffect, useState} from "react";
import AirbnbPriceMap from './AirbnbPriceMap'
import './App.css'
import Header from './Header'
import Footer from './Footer'
import underscore from 'underscore';
import compile from './compile'
import templates from './templates'
import baseName from './httpFunctions'
let templateNames = Object.keys(templates).slice(0, 12)
let templateContent = Object.values(templates).slice(0, 12)
let documentContext = {}

async function _() {
  let text = get('textarea').value.split('\n').map(_ => _.trim()).filter(_ => _)
  const url = `${baseName}/makeFn`
  console.log('url', url)
    let fn = await fetch(url, {
      method: 'POST',
      redirect: "follow", // manual, *follow, error
      referrerPolicy: "no-referrer", 
      mode: "cors", // no-cors, *cors, same-origin
      cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      headers: { 
        "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*"
    },
                body: JSON.stringify({fn:text,
                                      documentContext: getFormData(),
                                      hashUrl: window.location.hash.slice(1)
                })
    })
    fn = await fn.json()
    documentContext = fn.documentContext
    if (fn.isFromDisk) document.querySelector('textarea').value =  fn.fn
    return fn.fn 
}

function CodeEditor({apply_}) {
  useEffect(() => {
    get('textarea').value = templateContent[0]
    apply_()
  }, [])

  return (<><textarea 
  className="w-96 h-64 rounded-lg p-2 bg-black text-gray-400"
  onKeyUp={delay(apply_, 1000)}> 
  </textarea> 
  </>)
}

function App() {
  const [count, setCount] = useState(0)
  const [components, setComponents] = useState([])
  const [getSelect, setSelected] = useState(0)

  const apply_ = underscore.debounce(function () {
    async function apply_(){
      let data = await _()
      data = compile(data, apply_);
  
      setComponents(data)
    }
    apply_()
  }, 5 * 1000)
  
  useEffect(() => { 
    const fetchData = async () => {
      let data = await _()
      data = compile(data, apply_);
      setComponents(data)    
    }

    fetchData()
    
  }, [count])

  const leftPanel = (
    <div className="p-5 xs:hidden sm:hidden sm:hidden lg:block">
    <CodeEditor apply_={apply_}></CodeEditor>        
    <label className='block text-gray-500'>pick a template</label>
      <ul
      value={getSelect}
      className="w-96 m-5 text-gray-500">
      {templateNames.map((key, index) => 
      <li
      className="hover:bg-purple-500 truncate"
      style={{'color': index == count ? 'purple' : '' }}
      onClick={(e) => {
        get('textarea').value = templateContent[index]
        setCount(index)
        }  
      }
      key={key} value={index}>{1 + index} - {key}</li>)
      }
    </ul>
  </div>
  )
  
  return (
  <div className="overflow-scroll h-screen">
  <Header />
    <div className="grid grid-cols-4">
    {leftPanel}
      <div className="col-span-3">
        {components} 
        {/* <AirbnbPriceMap/> */}
        </div>
    </div>
    <Footer />
  </div>
  )
}

function delay (fn) {
  let one = Date.now()
  return function () {
    let two = Date.now()
    if (two - one > 2000) fn()
    one = two
  }
}

function setFormData (key, val) { documentContext[key] = val } 
function getFormData (key) { return key ? documentContext[key] : documentContext}
function get (query) {
  return document.querySelector(query)
}

export default App