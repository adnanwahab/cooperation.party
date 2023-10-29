
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
  return (<>
    <h3 className="">World Map - Scroll to zoom in to see every home in the world at a higher resolution</h3>
    <div className="relative h-96" style={{left: `${props.left}px`}}>
    <ColorLabels></ColorLabels>
    <DeckGL
        width={1200}
        height={600}
      layers={layers}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
      onViewStateChange={({viewState}) => {
        //console.log('viewState', viewState)
        fetchInterestingData(viewState)
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
    </>
  );
}

function fetchInterestingData(viewPort) {

  const viewport = new WebMercatorViewport({...viewPort});
  const topLeft = viewport.unproject([0, 0]);
  const bottomRight = viewport.unproject([viewport.width, viewport.height]);

  const boundingBox = {
    top: topLeft[1],
    left: topLeft[0],
    bottom: bottomRight[1],
    right: bottomRight[0]
  };
  console.log('bbox', boundingBox)
  console.log('spatial lite and get rainbow routes')
}



export default function AirbnbPriceMap (props){
  const data = []
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
    const fetchData = async () => {
        const data = [];
        for (let key of cityNames) {
          
        }
        const promises = cityNames.slice(0, 10).map(async city_name => {
            const req = await fetch(`https://shelbernstein.ngrok.io/data/airbnb/apt/${city_name}`);
            const json = await req.json();
            for (let key in json) {
                data.push(json[key]);
            }
            return data;
        });

        Promise.allSettled(promises).then(results => {
            const allData = [];
            results.forEach(result => {
                if (result.status === 'fulfilled') {
                    allData.push(...result.value);
                } else {
                    console.error("Error occurred:", result.reason);
                }
            });

            setCityData(allData);
        });
    };

    fetchData();
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