
import React, { useEffect, useState} from 'react';
import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {AmbientLight, PointLight, LightingEffect} from '@deck.gl/core';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';
import {GeoJsonLayer} from '@deck.gl/layers';
import {interpolatePurples} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries
import { Transition } from '@headlessui/react'
import { WebMercatorViewport } from '@deck.gl/core';


function ProgressBar (props) {
  let style = {width: `${props.percentage || 0}%`}
  //let [_, set_] = useState(0)

  // useEffect(function _() {
  //   setTimeout(() => {
  //     set_(_+1)

  //   }, 1000)
  // }, [_])

  return (<div className="relative top-0 progress-bar h-8 bg-blue-500" style={style}> </div>)
}

function processChunk(list) {
  return [list[2]?.replace(/\"/g,''), list[3]?.replace(/\"/g,'')]
          .map(parseFloat)
}

const all_city_locations = []

async function fetchData(setState) {
  const response = await fetch("https://shelbernstein.ngrok.io/osm_bbox");
  // If you sent the total as a header:
  const totalChunks = 1e6
  //parseInt(response.headers.get("X-Total-Chunks"));

  const reader = response.body.getReader();

  let receivedChunks = 0;

  reader.read().then(function recurse(shit){
    receivedChunks++
    let text = new TextDecoder().decode(shit.value);
    text = text.split('\n')
                .map(_ => _.split(','))
                .map(processChunk)
    setState(text)
    console.log(text)
    if (! shit.done)
    setTimeout(() => {
      reader.read().then(recurse)
    }, 1000)
  })

  // async function* readStream() {
  //     while (true) {
  //         const { done, value } = await reader.read();
  //         if (done) break;
  //         yield value;
  //     }
  // }

  // for await (let chunk of readStream()) {
  //     const text = new TextDecoder().decode(chunk);
  //     console.log('fart', text, totalChunks)
  //     const match = /data: (.*)\n\n/.exec(text);
  //     if (match) {
  //         const data = JSON.parse(match[1]);
          
  //         if (data.type === "init") {
  //             console.log(`Total chunks: ${data.total}`);
  //         } else if (data.type === "data") {
  //             receivedChunks++;
  //             const percentProgress = (receivedChunks / totalChunks) * 100;
  //             console.log(`Received chunk ${receivedChunks}. Progress: ${percentProgress}%`);
  //         }
  //     }
  // }
}

function AirbnbWorldMap(props) {
  let layers = [
    new ScatterplotLayer({
      id: 'scatterplot-layer',
      data:props.data,
      pickable: true,
      opacity: 0.8,
      stroked: true,
      filled: true,
      radiusScale: 6,
      radiusMinPixels: 10,
      radiusMaxPixels: 10,
      lineWidthMinPixels: 1,
      getPosition: d => [d[1][1], d[1][0]].map(parseFloat),
      onClick: ({object}) => {
        console.log(object)
        let url = `https://www.airbnb.com/rooms/${object[0]}`
        window.open(url)
      },
      //getPosition: d => centroid,
      getRadius: d => 10,
      getFillColor: d => [Math.random()* 255, 0, 255],
      getLineColor: d => [0, 0, 0]
    })
  ]

  let layer = new HexagonLayer({
    id: 'hexagon-layer',
    data: props.data,
    pickable: true,
    extruded: true,
    radius: 20000,
    radiusScale: 100,
    elevationScale: 4,
//    getPosition: d => d.slice(0, 2),
    getPosition: d => [d[1][1], d[1][0]].map(parseFloat),
  });
  //layers.push(layer)
  
  const INITIAL_VIEW_STATE = {
    longitude: 12,
    latitude: 29,
    zoom: 1.24,
    minZoom: 1,
    maxZoom: 17,
    pitch: 0,
    bearing: 0
  }   
  const [currentViewState, setViewState] = useState(computeBoundingBox(INITIAL_VIEW_STATE))
  return (<>
    <h3 className="">World Map - Scroll to zoom in to see every home in the world at a higher resolution</h3>
    <div className="relative" style={{left: `${props.left}px`, height: '600px'}}>
    <ColorLabels></ColorLabels>
    <ProgressBar percentage={Math.random()}></ProgressBar>
    <DeckGL
        width={1200}
        height={600}
      layers={layers}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
      onViewStateChange={({viewState}) => {
        setViewState(computeBoundingBox(viewState))
      }}
      // parameters={
      //   blendFunc: [GL.ONE, GL.ONE, GL.ONE, GL.ONE],
      //   depthTest: false
      // }
    >
      <Map 
       glOptions={{preserveDrawingBuffer: true}}
      reuseMaps mapLib={maplibregl} mapStyle={MAP_STYLE} preventStyleDiffing={true}>
      </Map>
    </DeckGL>

    </div>
    <div>
      <div>Latitude: {currentViewState.top.toPrecision(4)} , {currentViewState.bottom.toPrecision(4)}</div>
      <div>Longitude: {currentViewState.left.toPrecision(4)} , {currentViewState.right.toPrecision(4)}</div>
    </div>
    </>
  );
}
async function fetchInterestingData (min_lat, min_lng, max_lat, max_lng) {


  //SELECT * FROM points
  // WHERE amenity IS NOT NULL -- this condition filters for rows which represent some sort of amenity/POI
  // AND MbrWithin(geom, BuildMBR(xmin, ymin, xmax, ymax)); 
  const request = await fetch(`https://shelbernstein.ngrok.io/cityList`)
  const cityList = await request.json()
}

function computeBoundingBox(viewPort) {

  const viewport = new WebMercatorViewport({...viewPort});
  const topLeft = viewport.unproject([0, 0]);
  const bottomRight = viewport.unproject([viewport.width, viewport.height]);

  const boundingBox = {
    top: topLeft[1],
    left: topLeft[0],
    bottom: bottomRight[1],
    right: bottomRight[0]
  };
  //console.log('bbox', boundingBox)
  //console.log('spatial lite and get rainbow routes')
  return boundingBox
}



export default function AirbnbPriceMap (props){
  const [cityData, setCityData] = useState([]);
  const [cityNames, setCityNames] = useState([])

  useEffect(() => {
    const fetchData = async () => {
      const request = await fetch(`https://shelbernstein.ngrok.io/cityList`)
      const cityList = await request.json()
      setCityNames(cityList)
    }
    fetchData()
  }, [])

  useEffect(() => {
    const fetchData = async (setData) => {
      const CHUNK_SIZE = 10; // Number of requests to process at once, adjust as necessary
      const data = [];
      
      const fetchCity = async city_name => {
          await sleep(500);
          console.log('lol', city_name)
          const req = await fetch(`https://shelbernstein.ngrok.io/data/airbnb/apt/${city_name}`);
          const json = await req.json();
          return json;
      };
  
      for (let i = 0; i < cityNames.length; i += CHUNK_SIZE) {
          const chunk = cityNames.slice(i, i + CHUNK_SIZE);
          const promises = chunk.map(city_name => fetchCity(city_name));
          
          const results = await Promise.allSettled(promises);
          
          results.forEach(result => {
              if (result.status === 'fulfilled') {
                  for (let key in result.value) {
                      data.push(result.value[key]);
                      setData(data);
                  }
              } else {
                  console.error("Error occurred:", result.reason);
              }
          });
      }
  
  };  

    fetchData(setCityData);
}, [cityNames]);

const [isShowing, setIsShowing] = useState(true)
  return ( <>
        <Transition
        show={isShowing}
        enter="transition-opacity duration-75"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-150"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
     <AirbnbWorldMap data={cityData}></AirbnbWorldMap>
    </Transition>
  </>)
}
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

export const colorRange = [
  [1, 152, 189],
  [73, 227, 206],
  [216, 254, 181],
  [254, 237, 177],
  [254, 173, 84],
  [209, 55, 78]
];

function getTooltip(params) {
    if (! params.picked) return 
    return JSON.stringify(params.object, null, 2)
    return params.object.url
}
//hex colors
function ColorLabels() {
    return <></>
}
async function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}