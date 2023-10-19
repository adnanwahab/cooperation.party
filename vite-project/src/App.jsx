//goal - get developers to make components that are scriptable with natural language
//how to - make like 200 and then share 
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Favicon from "react-favicon";
//npm install react-favicon --save
import {Runtime, Inspector} from "@observablehq/runtime";
import notebook2 from "35ca09d7f1457ad3";
import Header from './Header'

import HexagonWorld from './hexagon_world';
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

import Tabs from './Tabs';

import TravelTimeMap from './TravelTimeMap'
import TableView from './TableView'

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
console.log('is this deployed')

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
if (window.location.hostname == 'cooperation.party' || window.location.hostname === 'https://hidden-river-3971.fly.dev') {
//   console.log('gesundheit', window.location.hostname)
//    //url = `https://hidden-river-3971.fly.dev/makeFn/`
//    url = 'https://180b-73-77-43-211.ngrok-free.app/makeFn'
  url = 'http://cooperation-party.fly.dev/makeFn'

  url = 'https://pypypy.ngrok.io/makeFn'
}
url = 'https://shelbernstein.ngrok.io/makeFn'

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
  className="bg-blue-200 w-full h-64 border border-gray-300 rounded-lg p-2"
  onKeyUp={delay(apply_, 1000)}> 
  </textarea> 
  </>)
}

const List = (list) => 
  (<ul className="overflow-scroll h-64">
    <li key="item font-white">{list.length}</li>
    {list.map((item, idx) => <li key={idx}>{item}</li>)}
  </ul>)

function consoleLog(event) {
  return console.log(event.target.value)
}

let vote_titles = [
  'Astrophysics', 'Condensed Matter', 'Mathematical Physics', 'Mesoscale and Nanoscale Physics'
]
let votes = [0,0,0,0] //swap with 
// fetch('http://localhost:8000/.netlify/functions/run-rpc', {

const components = {
  'poll': Poll,
  'List': List
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
  <div class="text-xl">{props.city}</div>
    <HexagonMap {...props}></HexagonMap>
    {[...Array(8).keys()].map(_ => <br key={_}/>)}
    {_}
  </>
}

const isGeoCoordinate = (pair) => {
  return Array.isArray(pair) && parseFloat(pair[0][0]) && parseFloat(pair[0][1])
}

function isIsochroney(datum) {
  return datum[0] && datum[0][0] && datum[0][0][1] && datum[0][0][1].type === 'node'
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

function ShitAss() {
  const notificationMethods = [
    'tech conferences', 'political conventions' , 'music festivals'
  ]  

  function replace_me(content) {
    document.querySelector('textarea').value = 
    
    document.querySelector('textarea')
    .value.replace('get a list of tech conferences or music festivals one per month.',
    `get a list of ${content} one per month.`
    )
  }

  return (
    <div>
      <label className="text-base font-semibold text-gray-900">Clarifications</label>
      <p className="text-sm text-gray-500">How do you prefer to receive notifications?</p>
      <fieldset className="mt-4">
        <legend className="sr-only">Notification method</legend>
        <div className="space-y-4">
          {notificationMethods.map((notificationMethod) => (
            <div key={notificationMethod} className="flex items-center" onClick={() => replace_me(notificationMethod)}>
              <input
                id={notificationMethod}
                name="notification-method"
                type="radio"
                defaultChecked={notificationMethod === 'email'}
                className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-600"
              />
              <label htmlFor={notificationMethod} className="ml-3 block text-sm font-medium leading-6 text-gray-900">
                {notificationMethod}
              </label>
            </div>
          ))}
        </div>
      </fieldset>
    </div>
  )
}



function compile (dataList, apply_) {
//  let [getSelected, setSelected] = useState('')
console.log(dataList)
  const result = dataList.map(function (datum, index) {
    if (datum.component === '<clarification>') {
      return <ShitAss />
    }

    if (datum.component === 'schedule') {
      return 'schedule'
    }

    if (datum.component === '<BarChart>') {
      return <div class="overflow-scroll h-96"><BarChart 
      setFormData={setFormData}
      apply_={apply_}
      callback={(_) => console.log(_)}
      data={datum.data}></BarChart></div>
    }

    if (datum.component === '<traveltimemap>') {
      console.log('completedemo', documentContext.city)
      return <TravelTimeMap data={datum.data}></TravelTimeMap>
    }

    if (datum.component === '<tableview>') {

      return (
        <>
      <>{"This apt is best because of these reasons."}</>
      <TableView {...datum} h3_hexes={datum.h3_hexes} data={datum.data}/>
      </>
      )
    }

    if (datum.component === '<Hexagonworld>') {
      return <HexagonWorld data={datum.data}/>
    }
    if (datum && datum.isochrone) {
      
      return <><div><Isochronemap  {...datum}></Isochronemap></div> </>
    }

    if (datum[0] && datum[0].isochrone) {
      
      return datum.map((datum, idx) =><> <Tabs></Tabs><div key={idx} ><Isochronemap  {...datum}></Isochronemap></div> </>)
    }
    if (datum[0] == '#') return <h1 className="text-xl">{datum}</h1>
    if (isIsochroney(datum)) {
      return <Map data={datum}></Map>
    }
    if (datum.component === '<slider>') {
      return <><Slider apply_={apply_}  label={datum.label}/></>
    }
    if (datum.component === '<Radio>') {
      return <Radio 
      key={Math.random()}
      apply_={apply_} 
        formDataKey={datum.key}
       cities={Array.isArray(datum.data) ? datum.data : datum.data[getFormData()['continent'] || 'Asia']}></Radio>
    }
    // if (typeof datum === 'object' && ! Array.isArray(datum)) { 
    //   return <Histogram data={Object.values(datum)}/>
    // }
    // if (isGeoCoordinate(datum)) {
    //   return MapTrees(datum)
    // }
    if (Array.isArray(datum)) return List(datum)
    // //if (datum === 'lots of cool polling data') return Poll()
    // if (datum === 'timeseries') {
    // }
    // if (datum === 'housing_intersection') {
    //     return <HousingIntersectionFinder />
    // }
    return <TextPresenter text={datum} />
  })

  return result.map((_, idx) => <div key={idx}>{_}</div>)
}

function TextPresenter(props) {
  return <div className=" border border-purple-500">
    {props.text}
  </div>
}


//  my schedule is i go to the office 3 x a week my office is at brookfield place i go to yoga 4x a week and i go to rock-climbing 2x a week and i visit the closest park everyday.

let templates = {
  //shelter
  find_best_deals_on_airbnb: `
  get a list of tech conferences or music festivals one per month.

  my schedule is bar 3x a week, college 3x a week, and pub 2x a week, food_court 7x a week, library 2x a week, car_wash 1x a week, prison 1x a week, mailroom 3x a week, vending_machine 14x a week.

  find best deals on airbnb that dont have crimes or other issues and then also print a diagnostic report that simulates living there for one month and train distance to my favorite places = (research_institute, bench).
  `,
  //shelter
  how_much_airbnb_revenue_will_your_house_get: `
  for every house in the world.
  predict how much revenue it would yield over 30 years.
  if i have 200k, which ones should i buy in new york.
  how soon would i get money back?
  make this work means progress bars + streaming intermediate result + 11 days left come back soon.
  
  should you buy this house? depends on x,y,z 
  


  `,
  //water is solved by energy
  //cooperation = OPEN PROBLEM - better information???
  //education = more fun to build cool shit ??? - find citations -
  //Ecology = how to do design arcologies
  //pollution = are thoughts pollution? why are some thoughts constructive?? visualize recycling
  //Energy = what are smart power grids? progress in battery tech: find papers, citations, data -> make diagrams
  //population = how can we have 100 billion peoeple -> make maps of aqua cities - find meme from twitter
  //rights = find papers 
  //health = find papers -> withregard, thestudio
  //Climate = how do weather predictions happen? einstein told his barber, waitbutwhy, xkcd, 
  //our existence on earth required work in olden times to be subsistent. 
  // remote_year_planning: `
  // for every city in ['Tokyo, Japan', 'Houston, Texas', 'Madrid, Spain']
  // find all apt within commute distance to coworking`,
  // plan_immigrant_housing_or_graduates: `
  // for every city in ['Tokyo, Japan', 'Houston, Texas', 'Madrid, Spain']
  // find 10 houses and each house is close to the residents favorite preferences. All like bar, half like research_institute, only 1 likes clinic (two people like restaurant, two people like library, two people like atm,  none of them like vending_machine and they all like bench but half like libraries and the other half prefer parking_space and some prefer bank while others prefer place_of_worship and some like disco and the others prefer country.) - they all want to be less than 90 min train distance to Sensō-ji`,


  where_to_build_new_houses_for_max_suitability: `
    given airbnb revenue + commute distance + 100k, where do i plant a new house to get most lifetime happiness out of my fam and so on.
  `,
  hexagon_world: `map of the future - all airbnbs + pois in the world
  barchart every airbnb in every city
  barchart every house_listing in every city
  barchart every poi in every city
  `,
  // airbnb: `for each continent
  // choose a city in each
  // find all airbnb in that city
  // i like library
  // i like bar
  // i like coffee
  // filter by 10 min train or drive to a library above 4 star
  // `,

  power_of_mushrooms: `find all courses that are related to mushrooms 
    plot how this number is going up
    get facts related to how mushrooms will improve us economy by 500 billion in next 3 years.

  `,

  arxiv_paper_digest: `find all papers on https://scholar.google.com/scholar?start=0&q=IPC&hl=en&as_sdt=0,44
  find papers which are good but not highly cited yet and find papers that may be highly cited in future 
  get all diagrams
  get all tables
  get sentences that are highly similar or support the same thesis 
  :poll astronomy, physics, infoTheory
      find all papers on arxiv relating to astronomy
      predict papers that will be highly cited
      or predict quality or relevance of paper that are too specific to have a lot of citations
      
      What are the most important problems in your field -> how would you work on them?.
      `,

  // twitch_comments: `get all twitch comments from ['zackrawrr', 'elajjaz', 'arthars']
  // then group them into topics
  // when more than 5 comments about win trading or food -> order a pizza
  // detect spikes in user activity by recording timestamp
  // `,
  pokemon: `given a favorite pokemon - select
  make best team that counters elite 4
  element counters
  
  find recent compettions -> diagram how metagame changes
  `,

  //conversation skills for the aspiring citizen scientist - 5th grade + people like communities, humor, and chill energy :)
  // `keenan+kel-humor-transcription`: 
  // `get transcript from https://www.youtube.com/watch?v=a02S4yHHKEw&ab_channel=AllThatOfficial
  //  find jokes that are duplicated in modern family
  //  `,
   tree_planting: `find all trees in nyc 
   plot on a map
  reccomend places to plant a tree based on h3
  visualize how many are of which species in a trees_histogram
  trees_map
  find a place to plant a new tree that is optimal


  render routes of a person and plant 1 million trees
  render a schedule for watering these trees 
  `,
//  `dota`:  `
//  make a dictionary/graph out of https://dota2.fandom.com/wiki/Category:Counters
//   should look like complements, counterThem, countersYou
//  given a team pick - pick 5 best team that counters it`,

  yt_humor_database: `
  for all keenan and kel
  get all transcript
  then find the jokes
  then categorize the jokes and the other context 
  `,

  sentiment_analysis_news_all_of_it: `
    people dont remember the stupid shit. they remember the good shit.
    the world is really 10 billion heroes recogniziing each other. 
    sci-fi fantasy illusory reality probablity manipulation is becoming common and standard. 
    what could work better than news? 
    why do people even like news?
    what if there was a song daily at lunch about coolest new thing that day
    more conferences.
    show how and why it varies -> levels of energy + spiral dynamics
  `,


  music_theory_colors: `
  7 colors
  7 tones
  whats else?
  `,

  healthy_buddhist_conversation_right_thought_goenka_transcript_plus_coolest_diagrams: `
    https://www.youtube.com/watch?v=cz7QHNvNFfA&list=PLPJVlVRVmhc4Z01fD57jbzycm9I6W054x
    dalle on all the things?
    find concepts that want to be visualized
    awareness = film strip of focus snap shots
    how does process of breath -> use regl.party/
    find research how does focusing on breath make mind more optimal
  `,

  make_educational_materials_that_studio_would_love: `
    audit thestudio.yoga
    use whisper on course materials
    make diagrams
    send to them.
  `,
  // community_building: `
  // get all groups from meetup
  // which ones are most popular
  // why are they cool? 
  // `,


  // yoga: `
  // get interesting research papers aboout mushrooms
  // `,



  // yoga: `
  // get interesting research papers aboout yoga
  // `,

  // creative_AI: `
  
  
  // `,

  BLS: `
      what skills are rising fastest
      parse job listings for every company
      chart listings over time for each company
      for each job
      by industry

      --- python ++
      --- webgl --
      --- unity/unreal ++ 
      how many jobs require math skills
      how can i demonstrate those
      how to use LLMs for Mathmagics - read kill math

      khan academy is amazing but too fucking boring
      i want notebooks generated for each concept -> with automagic code problems that solve world porblem

      link to chapter 0: fundamentals computed -> solve proofs by hand
      learn math with flash cards
      how to write mathematically sound code with 

      linear algebra, calculus (AuC for blood sugar estaimes for food)
      statistics + probability
  `, //    join bls with better shit


  idealism_is_more_effective_in_2023_because_progress: `
    long time ago idealism sounded cool but impractical
    but today people working together for a common goal can improve billions of lives 
    begginigngs of infity -> cite soruces 
  `,
  robot_revolution: `
  When will robots be cheaper and widely spread?

  when will i have robots to clean my house?

  When will i have a robot doctor? - 2 billion kids want this

  When will i have a robot comedian?

  When will i have rosie?

  When will i have mega man x?

  What kinds of labor can robots do?

  How many robots are built every year?

  How is that number changing?

  How many research papers are there for robotics?

  What is the intersection between LLMS and robots?

  What are the big problems in robots?

  How much venture capital is changing in robots?

  how many robot movies and books are there?

  use 5 emojis and make a chatbot.
  
  `,

  health_care: `
  What are the biggest costs in healthcare?

  what is the supply shortage on nurses, technicians, doctors?  

  How can health insurance and malpractice insturance be made cheaper by 
    diagnosis becoming more effective, precise, less wrong, earlier.

  How are next generation medicines better than previous ones?

  what technologies make next generation medicines possible?
  `,

resource_utilization_economics: `
  how many resources are there?
  how many people can the economy support?
  Number one: In 1945, corporations paid 50 percent of federal taxes.
   Now they pay about 5 percent. 
   Number two: in 1900, 90 percent of Americans were self-employed; now it’s about two percent.
   how many farmers markets are there?
   how much do gmos improve crop yields?
   how many farmers are there?
   how do resources get to places?
   What are the different steps in resource transformation from commodity to finished good?
`,


food_flavor_networks: `
What are all the different foods?
Why are some good for you?
what is a food scientist
what are the coolest jobs in the future? 

`,

is_coffee_good_for_you: `
i dont know
the invention of algorthims may be correlated with the invention of arabica coffee
90% of the population is reliant on caffiene
caffiene is a femme fatale - menno henselman
coffee was popularized during the industrial revolution.
my thesis is that mushroom 2x weekly microdoses + yoga = less mind wandering = less desire for coffee.
sleep researchers dont drink coffee. investigate matthew walker

`,


information_economics: `
  what are the coolest ideas in hal varian's information economics?
  What can data scientists do? (prediction, planning, ambiguity resolution, answer questions)
  How much can a good data scientist improve a system?
  What are the systems that underpin the foundation or structure of the world?
  State of Nature when? -> industrial revolution -> today
  What was lost in each step
  https://apenwarr.ca/log/
  Theres so many people rambling on the internet
  7 billion vectors in 7 billion directions. 
  But also 7 billion heroes working together to make a better world 
  What if those vectors were somewhat more aligned to have less duplication and more innovation?
  https://blakemasters.tumblr.com/peter-thiels-cs183-startup

  What is the investment thesis of ARK and what are the subject staircase ladder to each one?
  can 5th graders contribute research papers or convert research papers to cool products?
  does the world need more products? 
  What are the coolest 100 products of last year?
  IF you were to make a small italian restaurant, how many users would you need to be happy?
  How can you make your users happy to know you?
  What underserverd markets are there
  What gaps in the market can be identified
  how can the "market" be visualized" and understood
  Is the market efficent?
  What easy problems are solved?
  What difficult problems are most worth solving?
  How important is collaboration to solving the most difficult problems?
  http://www.paulgraham.com/greatwork.html - 
`,

video_game_industry: `
  How fast is the video game industry growing?
  What is popular content on twitch 
`,

distance_to_eutopia: `
  gapminder covers this but reproduce their results from "scratch"
  whos right, UN or gapminder? and how can we instrument the process ??
  9 areas of alan kay's paper -> that was awhiles ago. how can it be updated
  LIVING research papers kind of.
  take 5 research papers and then synthesize code and data 
  What data are missing????????
  What is data and where does come from

`,

  engineering_and_speed_of_light: `
    What are the all the back of the envelope calculations needed to be a good engineer.
    D_trace blog.
    Wouldnt it be cool if there was millions of open source government servers for communal use?
    What are some new back fo the enveolope calculations?
    What are the million specific intelligences that will be needed in 10-20 years?

  `,

  text: `
  most text on the internet is true - so gpt is mostly truthy.
  how can more text on the internet be even more true.
  text is a good encoding for reality into dense information.
  What is information > how does information help people help themselves.
  imagine the whole world is like sims. you can see progress bars for everyone.
  imagine you can see a redacted summary of eeryones daily activity on a map
  how does the brain encode directions, and requests and know that it will be good idea to execute them?
  if more people write more truthy sentenes on internet, will ai get smarter?
  if ai gets smarter, how will it help more people help themselves?
  `,


  visualize_value: `make cool shit everyday`,
  math: `what are coolest ideas in measurement by lockhart`,

  agriculture: `
    what if every farm had a database?
    every crop that was planted and yielded makes an update
    then every farm can see every other farm.
    then people who need what farm makes can read them and update them to eat them.
    make a sim city visualization -> what would happen if city plants a building in this area
    traffic visualization -> simulation = create predicted data for visualization
    take nyc taxi data -> use this to build function
    use function to build same data for every road in every other city adjusted by sliders / population density
  `,

  traffic_analysis: `lewis lehe`,
  solve_internet_toxicity: `what this? why this bad. 
    mushroom may now automatically make society better.
    its possible to manufacture billions of doses and distribute the cheaply.
    but a revolution in culture = everyone more happy - jk
    mushrooms can change your mind. make it better. but this lasts maybe a few days or months.
    to make it permanent and to make more happy everyone in your life
    create more happy within self -> cite everything you say and have evidence
    and use evidence for positive good. 
    cite pg + copy gwern.
  `,

  thesis_in_next_7_years: `
    code gen is super cool
      what will code gen be useful for? (engineer be more T shaped)
      the other 99% of the population = be better at computing 
      what are computers really useful for??

      are there codes that take 3 months of work today that can produced 100x faster? 

      generating images is great work. 
      generating living diagrams like bret victor may also be great work

      http://www.paulgraham.com/greatwork.html

      make codes that 100% of people want to edit,customize, understand, and audit.
      make documents that 10% of people want to work on together. - distil.pub = cool but some are a 5 man effort.
      how much effort does it take to make a cool things?
      What cool stuff need to be made?
      How many side projects are there on github?
      How many side projects are successful in the world?
      Maybe someday, everyone in the world can make and maintain a cool side project to be self reliant
      It would be cool to see a traversable network visualization of all these projects?
      now what some of the most important projects that humanity is doing or about to do. 
      what are some important projects that need to be done and no one is doing
      http://www.paulgraham.com/greatwork.html


    What are the greatest scientific break throughs of the last decade?
    List all nobel prize winners? predict next 100=
  `,

  appreciate_previous_1000_inventions_list_next_1000_inventions: `
    styofoam was invented during the space race and now it groceries cool while they travel in transit 
    should every person have their own garden?

  `,

  appreciate_good_writing: `
  - [ ] review notes and onvert to coop.party
- [ ] how much can good writing change the world - see paul graham - 600 billion dollars
- [ ] how much can good writing / journaling change your life / mind? open research topic
- [ ] now we can automagically add diagrams to our writing and tune that drawing dynamic images like bret victor.
  `

  //visualizing the mind 
  //minority report done more humanely -> instead of prison -> detect randomness at earliest onset
  //mushroom microdose, yoga and hike with eric.
  //connect to your humanity. problem solved. pdf printed august 10. april 5. october 5. december 5
  //an engineer or scientist can remove limitations from systems
  //life was difficult a long time ago.
  //now we have systems that support various aspects of our humanity.
  //design and creativity are just the combination of two things
  //heres the progress bars on important problems that are being solved by everyone in the 100 billion humans
  //everyone in the last 100 billion humans were heroes.
  //you can be like navi. you can clone 7 billion instances of yourself to guide the next 1000 billion humans in their future endeavors
  //war, famine, pestilience and whatever are obsolete, nothing but a memory.
  //in the next 7-20 years - find out how much - we can move the levers that will make everyone happy care bear wizard sci fi reality - Starcraft MMO yay
  //honor the sacrifices of the people who came before us. 
  //infinite party mode 5ever 2029 onwards 
//  healthcare: await('data/healthcare.txt')
}

let templateNames = Object.keys(templates);
let templateContent = Object.values(templates);

function App() {
  const [count, setCount] = useState(0)
  const [components, setComponents] = useState([])

  const [getSelect, setSelected] = useState(0)

  const apply_ = underscore.debounce(function () {
    async function apply_(){
      console.log('hello')
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
    <div>
    <CodeEditor apply_={apply_}></CodeEditor>        
    <label>pick a template</label>
      <ul
      value={getSelect}
      className="w-64 m-5 border border-bluegray-800 border-dashed">
      {templateNames.map((key, index) => 
      <li
      className="hover:bg-purple-500"
      style={{'border': index == count ? '1px solid purple' : '' }}
      onMouseOver={(e) => {
        get('textarea').value = templateContent[index]
        setCount(index)
        }  
      }
      key={key} value={index}>{key}</li>)
      }
    </ul>
    <Favicon url={cat} />
  </div>
  )
  
  return (
  <>
  <Header />
    <div className="grid grid-cols-3">
    {leftPanel}


      <div className="col-span-2 ">{components} </div>
      {/* <div>
      1. proximity to water
      2. land usage
      3. POI density by travel time
      4. parking
      5. pubs
      6. touristy things
      7. town hall
      8. distinctive buildings
      9. Atlas Obscura - cool things to visit
      10. tweet/review - get hipster/family-friendly
      11. light pollution - street lamp density
      </div> */}
    </div>
    <Footer />
</>
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