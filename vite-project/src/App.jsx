//goal - get developers to make components that are scriptable with natural language
//how to - make like 200 and then share 
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Favicon from "react-favicon";
//npm install react-favicon --save
import {Runtime, Inspector} from "@observablehq/runtime";
//import notebook2 from "35ca09d7f1457ad3";
import Header from './Header'

import Footer from './Footer'
//import notebook from "3f6e237936d0a0c7";
// import { MapContainer } from 'react-leaflet/MapContainer'
// import { TileLayer } from 'react-leaflet/TileLayer'
// import { useMap } from 'react-leaflet/hooks'
import { MapContainer, TileLayer, useMap, Marker, Popup } from 'react-leaflet'
import underscore from 'underscore';
import MapComponent from './Map'
import * as d3 from 'd3'
import barchartNotebook from "https://api.observablehq.com/@d3/bar-chart-race-explained.js?v=3";
import HexagonMap from './HexagonMap' //commute and so on

import BarChart from './BarChart';

import TravelTimeMap from './TravelTimeMap'
import compile from './compile'
import templates from './templates'
//1 million appartments -> 10  million routes -> 1-3 weeks to compute
//written as much code as you can for this setup
//to actually complete this app = deploy it desktop and upgrade the machine by 10x 


let templateNames = Object.keys(templates).slice(0, 12)
let templateContent = Object.values(templates).slice(0, 12)
//pick optimal housing location for next 10-30 years 
//visualize school disticts
//visualize proximity to closest whole foods
//visualize proximity to biking path
//print 5 best choices -> visit them 
//map scatterplot trees.csv
//map isochrone union_square
//map heatmap water_fountains
//select regions on above 3 maps and then sort by most likely to appreciate in value
//return list of houses 
function Slider (props) {
  let _ = {}
  _[props.label] = .5
  return <div className="block">
  <label>{props.label}</label>
  <input type="range" onChange={(e) => {
      _[props.label] = e.target.value / 100;
      setFormData('sliders', Object.assign(getFormData('sliders'), _))
      props.apply_()
      }
  } />
  </div>
}


let documentContext = {}
Object.defineProperties({}, {
  set: function () {
    //send document context to server using _
  }
})

function setFormData (key, val) {
  documentContext[key] = val
} 

function getFormData (key) {
//  if (! formData['city'])  return Object.assign(formData, {'city': 'Tokyo, Japan'})
  return key ? documentContext[key] : documentContext
}

let percent = 0
function ProgressBar () {
  let percentage = percent++
  let style = {width: `${percentage}%`}
  let [_, set_] = useState(0)

  useEffect(function _() {
    setTimeout(() => {
      set_(_+1)

    }, 1000)
  }, [_])

  return (<div className="hidden absolute top-0 progress-bar h-8 bg-blue-500" style={style}> </div>)
}


function isNestedArray (arr) {  
  return Array.isArray(arr['Asia']) 
}

let hasBeenFlagged = false
function Radio(props) {

  let cities = props.cities

  let default_value = ''

  if (! hasBeenFlagged) {
    if (cities.indexOf('Tokyo, Japan') == -1) return 
    hasBeenFlagged = true
    default_value = 'Tokyo, Japan'
  }

  let [getCity, setCity] = useState(default_value)
  let [_, set_] = useState(false)

  let apply_ = props.apply_

  useEffect(() => {
    setFormData(props.formDataKey, getCity)
    apply_() //dont just send text -> send the data on the client 
    //get all satellite images for every country in america
    //tune contrast -> see if there are any patterns
  }, [getCity])

  

  return (
    <div>
      <fieldset className="mt-4">
        <div className="space-y-4">
          {cities.map((city, idx) => (
            <div key={idx} className="flex items-center">
              <input
                id={city}
                name="notification-method"
                type="radio"
                checked={getCity === city}
                className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-600"
                onChange={() => {setCity(city)}}
              />
              <label htmlFor={city} className="ml-3 block text-sm font-medium leading-6 text-gray-900">
                {city}
              </label>
            </div>
          ))}
        </div>
      </fieldset>
      -----------------------------------------------------
    </div>
  )
}

//import notebook3 from "@groundbear/housing123";

// let houses = await d3.csv('/calif_housing.csv')
// console.log(houses)

function HousingIntersectionFinder() {
  const ref = useRef();

  useEffect(() => {
    const runtime = new Runtime();
    let module0 = runtime.module(notebook3, Inspector.into(ref.current));
    module0.redefine('redefineColor', 'green');
    module0.redefine('houses', houses);
    module0.value('housesWithinIntersction').then((_) => { return console.log(_)})
    window._ = module0;
    //.define(["foo"], foo => `Hello, ${foo}.`)
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={ref} />
      <p>Credit: <a href="https://observablehq.com/d/ab80bfd4252083e3@95">Untitled by awahab</a></p>
    </>
  );
}




function Map(trees)  {
return MapComponent(trees)

return (
  <MapContainer center={[51.505, -0.09]} zoom={13} scrollWheelZoom={false}>
  <TileLayer
    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  />
  <Marker position={[51.505, -0.09]}>
    <Popup>
      A pretty CSS3 popup. <br /> Easily customizable.
    </Popup>
  </Marker>
</MapContainer>
)
  trees = trees.slice(0, 1e3)
  const markers = trees.map((_, i) => (<Marker key={i} position={_}>
    <Popup>
      A pretty CSS3 popup. <br /> Easily customizable.
    </Popup>
  </Marker>))

  return ( 
  <div className="w-48 h-64 relative">
  <MapContainer center={trees[0]} zoom={13} scrollWheelZoom={false}>
    <TileLayer
attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
/>

  </MapContainer>
  </div>
  )
}


function Notebook() {
  const ref = useRef();

  useEffect(() => {
    const runtime = new Runtime();
    runtime.module(notebook, Inspector.into(ref.current));
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={ref} />
      <p>Credit: <a href="https://observablehq.com/d/3f6e237936d0a0c7">Collaborative Travel Schedule Generation for Conferences by awahab</a></p>
    </>
  );
}

// import notebook from "@sandraviz/animated-line-chart";
//import d3 from 'd3'

//import notebook from "@rithwikanand/histogram";

// const LASTDAY= function () {
//   console.log(LASTDAY)
//     return (
//      <>
//      <div id="observablehq-lineChart-c1e4ccc8"></div>
//       <p>Credit: <a href="https://observablehq.com/d/1420c244c74cb1bb">GNSS Position Timeseries by Yang</a></p>
//     <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@observablehq/inspector@5/dist/inspector.css">
//     <script type="module">
//   import {Runtime, Inspector} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/dist/runtime.js";
//   import define from "https://api.observablehq.com/d/1420c244c74cb1bb.js?v=3";
//   new Runtime().module(define, name => {
//     if (name === "lineChart") return new Inspector(document.querySelector("#observablehq-lineChart-c1e4ccc8"));
//   });
//   </script>
//   </>)
// }

function vennDiagram() {
  return (<svg id="vennDiagram" width="500" height="500">
        <circle cx="150" cy="250" r="100" style="fill:rgba(255,0,0,0.5);" stroke="#333"></circle>
        <circle cx="300" cy="250" r="100" style="fill:rgba(0,0,255,0.5);" stroke="#333"></circle>
    </svg>)
}

// import notebook4 from "6c8bcf8857e3482e";

// function Histogram(props) {
//   const chartRef = useRef();

//   useEffect(() => {
//     const runtime = new Runtime();
//     runtime.module(notebook4, name => {
//       if (name === "chart") return new Inspector(chartRef.current);
//     }).redefine('data', props.data)
//     return () => runtime.dispose();
//   }, []);

//   return (
//     <>
//       <div className="w-128" ref={chartRef} />
//     </>
//   );
// }
//<iframe width="100%" height="1004" frameborder="0"
//src="https://observablehq.com/embed/@ankitak/stars?cells=chart"></iframe>

import React, {useRef, useEffect} from "react";
//import notebook from "1420c244c74cb1bb";

// function Notebook() {
//   const lineChartRef = useRef();

//   useEffect(() => {
//     console.log('TIMESERIES')

//     const runtime = new Runtime();
//     runtime.module(notebook, name => {
//       if (name === "lineChart") return new Inspector(lineChartRef.current);
//     });
//     return () => runtime.dispose();
//   }, []);

//   return (
//     <>
//       <div ref={lineChartRef} />
//       <p>Credit: <a href="https://observablehq.com/d/1420c244c74cb1bb">GNSS Position Timeseries by Yang</a></p>
//     </>
//   );
// }

const diagrams = {
//  'histogram': Histogram,
  'timeSeries': Notebook
}

// `<iframe width="100%" height="584" frameborder="0"
//   src="https://observablehq.com/embed/@rithwikanand/histogram?cells=chart"></iframe>`

function get (query) {
  return document.querySelector(query)
}

// function MosaicCrossFilterFlightsM() {
//   const viewofFlightsRef = useRef();

//   useEffect(() => {
//     const runtime = new Runtime();
//     // const module0 = runtime.module(notebook, name => {
//     //   if (name === "viewof flights") return new Inspector(viewofFlightsRef.current);
//     // }).redefine('redefineColor', 'purple')
//     return () => runtime.dispose();
//   }, []);

//   return (
//     <>
//       <div ref={viewofFlightsRef} />
//       <p>Credit: <a href="https://observablehq.com/@uwdata/mosaic-cross-filter-flights-10m">Mosaic Cross-Filter Flights 10M by UW Interactive Data Lab</a></p>
//     </>
//   );
// }
//console.log('is this deployed')

async function _() {
  let text = get('textarea').value.split('\n').map(_ => _.trim()).filter(_ => _) //TODO whitespace removal
  //text = ['asdfasd', 'asdfasdf', 'asdf']
//  let port = 8000
  let port = 8000
//  let url = `http://localhost:${port}/makeFn/`
  let url = `http://127.0.0.1:${port}/makeFn`
//   let useGPU = true
//   console.log('gesundheit')
//    url = `https://pypypy.ngrok.io/makeFn/`
//    //url = `https://hidden-river-3971.fly.dev/makeFn/`
if (window.location.hostname == 'cooperation.party') {
//   console.log('gesundheit', window.location.hostname)
//    //url = `https://hidden-river-3971.fly.dev/makeFn/`
//    url = 'https://180b-73-77-43-211.ngrok-free.app/makeFn'
  url = 'http://cooperation-party.fly.dev/makeFn'

  url = 'https://pypypy.ngrok.io/makeFn'
} else if (window.location.hostname === 'localhost:5173') {

}


url = 'https://shelbernstein.ngrok.io/makeFn'
//on MBP localhost
//console.log('mbp', url === 'https://shelbernstein.ngrok.io/makeFn')
//console.log('prod+fly.io', url === 'https://pypypy.ngrok.io/makeFn')
// url = 'https://pypypy.ngrok.io/makeFn'

  // let fn_ = await fetch('mockData.json');
  // fn_ = await fn_.json();
  // console.log(fn_);
  // return {fn: fn_}
    let fn = await fetch( url , {
      method: 'POST',
      redirect: "follow", // manual, *follow, error
      referrerPolicy: "no-referrer", 
      mode: "cors", // no-cors, *cors, same-origin
      cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      headers: { 
        "Content-Type": "application/json",
      //"ngrok-skip-browser-warning": true,
      "Access-Control-Allow-Origin": "*"
    },
                body: JSON.stringify({fn:text,
                                      documentContext: getFormData(),
                                      hashUrl: window.location.hash.slice(1)
                })
    })
    fn = await fn.json() //
    documentContext = fn.documentContext
    if (fn.isFromDisk) document.querySelector('textarea').value =  fn.fn
    return fn.fn //fn return vals + components to render them


  // let fn2 = await fetch(url, {
  //   method: 'POST',
  //   headers: { "Content-Type": "application/json"},
  //             body: JSON.stringify({fn:text})
  // })
  // fn2 = await fn2.json()
}

function delay (fn) {
  let one = Date.now()
  return function () {
    let two = Date.now()
    if (two - one > 2000) fn()
    one = two
  }
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

let vote_titles = [
  'Astrophysics', 'Condensed Matter', 'Mathematical Physics', 'Mesoscale and Nanoscale Physics'
]
let votes = [0,0,0,0] //swap with 
// fetch('http://localhost:8000/.netlify/functions/run-rpc', {

const components = {
  'poll': Poll,
  'List': function () {}
}

function Notebook2() {
  const chartRef = useRef();

  useEffect(() => {
    const runtime = new Runtime();
    runtime.module(notebook2, name => {
      if (name === "chart") return new Inspector(chartRef.current);
    })
    // .define("houses", houses)
    // .import("houses", "houses", houses)
    // .variable().define("houses", houses)
    .redefine('houses', houses)
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={chartRef} />
      <p>Credit: <a href="https://observablehq.com/d/35ca09d7f1457ad3">Learn D3: Animation by awahab</a></p>
    </>
  );
}

function Isochronemap(props) {
  if (! props.reports) return <>not found sorry 🐻</>
  console.log(props)
  let _ = props.reports.map((_, idx) => {
    const reasons = _['reasoning_explanation']
    return <div key={idx}>
            <div>{_['name'] + '  ' + _.house.url}</div>
            <>Other Houses Within Neighborhood{List(_.houses_within_suggested_neighborhood.map(_ => _.url))}</>
            <div>{reasons.split('\n').map(_ => <div>{_}</div>)}</div> 
            <div>
               <BarChart data={Object.entries(_['commutes']).map(_ => {
                  return { letter: _[0], frequency: parseFloat(_[1].replace('mi', '')) }
                })}></BarChart>
            </div>
          </div>
  })
  return <>
  <div className="text-xl">{props.city}</div>
    <HexagonMap {...props}></HexagonMap>
    {[...Array(8).keys()].map(_ => <br key={_}/>)}
    {_}
  </>
}

const isGeoCoordinate = (pair) => {
  return Array.isArray(pair) && parseFloat(pair[0][0]) && parseFloat(pair[0][1])
}


//list airbnbs in each city -> bar chart
//find all that are commute distance to best coworking space in each one. 


//
//

//match sentence with presenter on client
//fetch model with fetch from server
function SentencePresenter() {
  let originalSentence = ''
  return (<>
    {originalSentence}
  
  </>)
}

/*
  This example requires some changes to your config:
  
  ```
  // tailwind.config.js
  module.exports = {
    // ...
    plugins: [
      // ...
      require('@tailwindcss/forms'),
    ],
  }
  ```
*/
const notificationMethods = [
  { id: 'email', title: 'Email' },
  { id: 'sms', title: 'Phone (SMS)' },
  { id: 'push', title: 'Push notification' },
]





//government 
//make it so 300 million people can talk to 300 million people
//in a constructive manner once a week -> 




//  my schedule is i go to the office 3 x a week my office is at brookfield place i go to yoga 4x a week and i go to rock-climbing 2x a week and i visit the closest park everyday.


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
    {/* <Favicon url={cat} /> */}
  </div>
  )
  
  return (
  <div className="overflow-scroll h-screen">
  <Header />
    <div className="grid grid-cols-4">
    {leftPanel}

      <div className="col-span-3">{components} </div>
  
    </div>
    <Footer />
  </div>
  )
}

export default App

let cat =  [
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQ8YIQwbLgQeNxMXMBMeMh0XOR4fMQ0gPhUiNRUnOyMcOyEpLCEvMCQ+PTgzMAIpSgYqTgcnUAAuWgUuXAorVAsuVAgvVg4rUw0qVwwtUAwvWQ4yTAYzVAEwXgAwXwEyXQAyXgA1WwM0XAE0XwE3XQI3XwcwXgY1XwA4XgoxWgswXAk2Xw4xWQ04WxMpQRUpRBMoVxEtVRUrVBUtWREyXRc5VQAvYAAvYQAsZwAuZwAxYAAwYQExYgAwYwIyYAAyYQAyYwAwZQAxZgEzZAAyZQAyZwE0YQE0YgA0YwE2YwA1ZAI0ZQA0ZgE0ZwA3ZAA3ZQA2ZwQwYQYyYwQyZAUzZQY1YQQ0YgY0Zgc2ZAQ3ZgIxaQAxagEzaAAyaQAyawAzbAAybgA0aAA0aQA1awE2aAI3aQA3agM0bgE2bAE3bQA2bwQwbwc3aAQ3bAU2bgA5YwQ4YAU4ZwU6ZgA4aAM4bAQ4aAQ5awgyZgwyYQ40ZQg4aAA1cAA1cRMpYhQrYRUrZRcsYyMuTi84SSg/XwpWeBlXcCpDWTFDUShTZDVAYFNIG2pOAGFdE05RWE5VXVRKbFRdZExnVltoeXxLT2xDZWNHcHpYc25haG5gbXJveHpueHV2eAtehnFWg3lUh3tWinxXjXpag25jg2Z1iHJihmWXlGmbkn+OpYJZAIRfV4NGf4tOd4FTb5BbiZh9jqZ2nqxxoauocd/OduXKb4aLkYiDmpOEl5KQk5iTl5GVoIi1oa6HnKOTnrWbpMCovtaQzt+W29+f09Soy9qt1N+41+WYxuKd1uWZ1+yc2+yV4/GR6PqY7fmd6v6b6+mszeal2+Wr1OC23uO53euq4PCm4/Sm4Pqh8dLVhODE0wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJNhIKMAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjVJivzgAAABG0lEQVQoUwEQAe/+AHxhJSFMaUVIR05zTVNURGEAe2MjcDtBYm1KSkBNNmw9UABdXDxTUlduOVt0TUtDQEpIAE09XHcYM35/fYAwGCpVSkoAZGJoE4GinZ+gnqGkiRkdZQA7QEQ0j9LP1MbB0MS1LylGAHkmKzKV08nWwKqwwraCNRwACwwNAJTVy8q/ra+WmLmGGwCoi4yKk8XNzKu7rry4tKMuALOy17GpzsjHrL2Xt5uQpwcApaa6kQ7Yw9G+mpmNjpKDMQCchIWIhwYKBQMECAEJAhR4AChvP1EaFxEsKRYVEA8tJz4ASU9eOHc8WTphYTZ6HxJyPwBnQnVJElg8aV5qYzd2O3FHAGtaICQiI0RfYHtfXWZEHlYX2mbiHrA6rAAAAABJRU5ErkJggg==',
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQ0dPhQXKBYYPw0kNgghPxInOhAmPRIoPx0mNTcfNTgyGjAlNg4aQhkbSwQoSAcqSAYqTgQvTwksSAUrUgcuVwAuWgEtXgAuXQAvXgAuXwYqXgQtWQsuWAguWw0uWQwuXgMzVwAwWAAxXQIzXAAzXgE1XAM0XQA0XgM0XwE2XgM3XwczWgQzXwY0WAU1XQsyUw0wUAoxWAsyWwsxXgkyXgwzXBEqSRIvTRAvThYoThcrThQsThgjQRwlTBspThIsURAtWRAsWxAvXhstURQwVxAwWRY4VAAvYgAtZgEuZgAvbggrbwIneAAxYQEyYAAyYQAyYwAwZQAwZgAxZwEzZAAyZQIyZgAyZwA0YAI1YgA0YwI3YQA2YwA1ZAE1ZQA0ZgE0ZwA2ZQA2ZgczYgYyYwYxZAUzZAYyZQQyZgQzZwU1ZgY2ZwAxawIyaAAzaQAyawAzbQA0aAA0agE2aAQ2awU0bAI5YgE4Zwc4YQY5ZAA4awM7bgU4aQQ6aQQ8bQk2YAg0ZQk1ZgM0dgM6cgk3cgwxfy82QCw4RjE8WipEXS1JVzRHTiVBZihPbC9SaCZRc0VINVVOMVNePlp2OWtaPkNQWEtUWV5OdVxbYVpib3NPWWtHY2hIbWRdZXhbb3tjS3dmcHl3eGVdhmpeiW1ciGRiimpkimpmiWR0jW91gXtthGqlVWipWYdPfoxSeo1BgpFjipJ6iqB5mLhpqoO0Y52xabPge7ThfIuJlJKCjZqPlZGVmJyTmISKopaMpZKUoKeJobibr7jigM+QyduYwd6c48Sgxt2i0t2h4eWT3e6U1POY0++T6PGT5/SZ6fqc6vqf7v2f9+es2uqm1+O/y/Cyx+Gg5Omj6eit4+un8OW24uSx6fOt4QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJj3aiYAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjVJivzgAAABG0lEQVQoUwEQAe/+AHF3IVtvcHJaJyNjZGVpUF4AYE92JGdnTi4rMhwzZhl8TwBifk8sRT84Nzs6Q0QVfRdZAF9HczQ9p6aio6OkpYwteCQAg29TQpfayNnF1tfbvgURKQBBHjU5nNPO0cmvs8exhkYgAAkLhwGb3M3Mw7CynqC/iS8AlJGSkJrEz9CtwLm8urioNgC2t8K1n9XLyq7BnbuhmL08AKyrtJMK1NLYxqqplZaZiD4Aj42OiosIAg0MAAYDBwRAHwBMhUsaMBITHRsxDg8QFIBWAIJKSHRnFoFrfBh5KGpfTlwAbnt6VmyESVdxTV9fcVVhXABYIlBVbVJ1bV13UVUXWWFiACUqVGBoSXVSUGFXVn8mTXr3Hmu6vgXxNAAAAABJRU5ErkJggg==',
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQkbLAEdOQAfPRwcMRIgLh4kKiAhIyQmKTcxIQ0fTwEhQQAqUwArVAAsVgAtWQAuWwAvXA8wUAwxVhwzSQAxYAAyYgAyYwAzZAAzZQA0Zik4Riw5RjM7Ti9AUjhCSzhDW3ZqOElGQkhISFxRRlxUTFFRUVJSUl9ZWWFWSWhWRGZdXWlWaXZYdnhlVHlgX3lkWGVkZHNlaH1uaXN0dXVzc3R0dHh4eDY6vzgzuTk3wCWMPBnGixfBlkHTlpgyCI5Ke7J8fc8wE4BVgKh4gq5vrKiDbcG0Ad/jAefeAfnVEoeFh4qKipKSkpSUlJmZmcuZmdubsPi9ot6F3u+P7/6G6v6G6/+G7P6J7v+L8P+L8f+O9P+P9f+R9/+T+f+U+f+X/P6d8f+Y/P+Y/v+Z//+a/vmkx/mrzv6uzP+m5QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE3XPx0AAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjVJivzgAAAAmklEQVQoU2OQRAMMWAWEdPVAQF9WQlISrEIlPTopLDw5w5UfJpAYGx+s7RLhLC0MVCEoxaicEhUZZOqlY+grJ8ngaKehqBkSl2Tv6+3n623My+Dh7ummkBAT6uSv5WOm7sDNYG1ja8URmJoWYGSipmogI85gaWGuJM/KzMnEwsbAzgW2RYJbVIBPUlKER4xbHGotsmuxO500FQCjIiU3D5OSIgAAAABJRU5ErkJggg==',
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURQ0dPhQXKBYYPw0kNgghPxInOhAmPRIoPx0mNTcfNTgyGjAlNg4aQhkbSwQoSAcqSAYqTgQvTwksSAUrUgcuVwAuWgEtXgAuXQAvXgAuXwYqXgQtWQsuWAguWw0uWQwuXgMzVwAwWAAxXQIzXAAzXgE1XAM0XQA0XgM0XwE2XgM3XwczWgQzXwY0WAU1XQsyUw0wUAoxWAsyWwsxXgkyXgwzXBEqSRIvTRAvThYoThcrThQsThgjQRwlTBspThIsURAtWRAsWxAvXhstURQwVxAwWRY4VAAvYgAtZgEuZgAvbggrbwIneAAxYQEyYAAyYQAyYwAwZQAwZgAxZwEzZAAyZQIyZgAyZwA0YAI1YgA0YwI3YQA2YwA1ZAE1ZQA0ZgE0ZwA2ZQA2ZgczYgYyYwYxZAUzZAYyZQQyZgQzZwU1ZgY2ZwAxawIyaAAzaQAyawAzbQA0aAA0agE2aAQ2awU0bAI5YgE4Zwc4YQY5ZAA4awM7bgU4aQQ6aQQ8bQk2YAg0ZQk1ZgM0dgM6cgk3cgwxfy82QCw4RjE8WipEXS1JVzRHTiVBZihPbC9SaCZRc0VINVVOMVNePlp2OWtaPkNQWEtUWV5OdVxbYVpib3NPWWtHY2hIbWRdZXhbb3tjS3dmcHl3eGVdhmpeiW1ciGRiimpkimpmiWR0jW91gXtthGqlVWipWYdPfoxSeo1BgpFjipJ6iqB5mLhpqoO0Y52xabPge7ThfIuJlJKCjZqPlZGVmJyTmISKopaMpZKUoKeJobibr7jigM+QyduYwd6c48Sgxt2i0t2h4eWT3e6U1POY0++T6PGT5/SZ6fqc6vqf7v2f9+es2uqm1+O/y/Cyx+Gg5Omj6eit4+un8OW24uSx6fOt4QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJj3aiYAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjVJivzgAAABG0lEQVQoUwEQAe/+AHF3IVtvcHJaJyNjZGVpUF4AYE92JGdnTi4rMhwzZhl8TwBifk8sRT84Nzs6Q0QVfRdZAF9HczQ9p6aio6OkpYwteCQAg29TQpfayNnF1tfbvgURKQBBHjU5nNPO0cmvs8exhkYgAAkLhwGb3M3Mw7CynqC/iS8AlJGSkJrEz9CtwLm8urioNgC2t8K1n9XLyq7BnbuhmL08AKyrtJMK1NLYxqqplZaZiD4Aj42OiosIAg0MAAYDBwRAHwBMhUsaMBITHRsxDg8QFIBWAIJKSHRnFoFrfBh5KGpfTlwAbnt6VmyESVdxTV9fcVVhXABYIlBVbVJ1bV13UVUXWWFiACUqVGBoSXVSUGFXVn8mTXr3Hmu6vgXxNAAAAABJRU5ErkJggg==',
]

function Poll() {
  const [voted, setHasVoted] = useState(false)
  let buttons = <>
      <button onClick={() => vote(0)}>
           {vote_titles[0]}
        </button>
        <button onClick={() => vote(1)}>
        {vote_titles[1]}
        </button>
        <button onClick={() => vote(2)}>
        {vote_titles[2]}
        </button>
        <button onClick={() => vote(3)}>
        {vote_titles[3]}
        </button>
  </>
  let graph = (<iframe width="100%" height="584" frameborder="0"
  src="https://observablehq.com/embed/@d3/bar-chart?cells=chart"></iframe>)

  function vote ( ){
    // setHasVoted(true)
  }
  return  <> {voted ? graph : buttons}</>
}

//choose your own adventure

//poll
//go to place -> geohashing 
//make pokemon go
///find a list of all pokemon
//for each pokemon -> list counters
//for each counter -> list all possible technique combinations 
//given 10 different teams of 6, estimate probability of them winning 
//do a "simulation" of 10 teams in a round robin and then find the best win rate


//twitter -> watch for pizza -> stream sentiment analyis 
//stream sentiment analysis for region
//if you see a spike or anomoly -> visit that region
//confirm whether bread or cheese supply has gone down 
//invest in new farm to produce better bread
//profit 

//make it so the python async functions return value with a generator 
//this generator steps through the tiem series paginates 1billion tweets 
//in a 10,000 item page so that the javascript component renders realtime 

//jupyter cant really do this -> cant server files from jupyter
//javascript cant exec on files that cant fit in browser which is lots 
//javascript max = 200mb -> python services can upload 200mb in chunks so it streams 40x faster so you can do proper analysis 





//what can you do with a mix of python and javascript that you cant do with one alone
//english -> generate javascript UI component -> generate python service that generates data and then ui co


//program synthesis -> make that better 

//given two subjects on arxiv -> find cool papers that relate to both of them

//use a IDE to think of things you would not have thought of 
//from a sentence -> generate a program
//from a program -> generate an a comment
//from a program -> generate a data 
//from a data -> generate helper functions to prep that data for applying statistics
//from a statistics output, you can exectue a program in ways that affect stfuf
//exampel -> reccomendation engine -> generates links on page
//from a list of links -> generate intersection or pairwise stuff

//list of favorite authors -> find common themes
//from a theme -> reccomend movies 


//5 people take a quiz that comes from a document describing questions

//depending on their ansewr they plan a movie + game night w/ apple cider vinegar and popcorn
//ok watch this movie 


//the more you type -> the more the autocomplete assistant knows exactly what you mean -> or even things you might like -> things you might not know
//as you type it helps you think -> renders shit on the right hand panel

//write stuff -> computer tells you if it makes sense
//if it thinks it doesnt make sense it tell you why

//write a note that has live updating sentences
//and you have diagrams that are UI components that can change the sentences
//you have a list of sentences
//ask LLM is this statement true
//sentences which evaluate to 

//look up all git hub repos
//find all books on gutenberg project
//find all events on gdelt

//transcribe all audio on youtube

//is this idea good?
//poll people 
//if its good -> add tech specs

//create tech spec
//generate tests
//tests are assertions that run on previous lines of code

//english  = glue language 
//linked diagrams
//javascript -> display + run queries on UI
//python -> automatically create services on infinte data streams 