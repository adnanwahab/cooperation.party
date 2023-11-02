import DonutChart from "./DonutChart"
import Histogram from './Histogram'
import React, { useEffect, useState, useCallback, useRef} from 'react';
import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';
import {GeoJsonLayer} from '@deck.gl/layers';
import {interpolateRainbow} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries
import { Transition } from '@headlessui/react'
import { WebMercatorViewport } from '@deck.gl/core';
import {DataFilterExtension} from '@deck.gl/extensions';
import {IconLayer} from '@deck.gl/layers';
import _ from 'underscore'
import * as turf from '@turf/turf';
import PopOver from './PopOver'
import {isWebGL2} from '@luma.gl/core';
import {ScreenGridLayer} from '@deck.gl/aggregation-layers';
import baseName from './httpFunctions'

const colorRange = [
  [255, 255, 178, 25],
  [254, 217, 118, 85],
  [254, 178, 76, 127],
  [253, 141, 60, 170],
  [240, 59, 32, 212],
  [189, 0, 38, 255]
];


const ICON_MAPPING = {
  marker: {x: 0, y: 0, width: 128, height: 128, mask: true}
};
const INITIAL_VIEW_STATE = {
  longitude: 12,
  latitude: 29,
  zoom: 1.24,
  minZoom: 1,
  maxZoom: 17,
  pitch: 0,
  bearing: 0
}   
// INITIAL_VIEW_STATE.longitude = 139
// INITIAL_VIEW_STATE.latitude = 35
// INITIAL_VIEW_STATE.zoom = 10

function AirbnbWorldMap(props) {
  const [routes, setRoutes] = useState([])
  const [markers, setMarkers] = useState([])
  const [getPercent, setPercent] = useState(0)
  const [openPopover, setOpenPopover] = useState(false)
  const [currentViewState, setViewState] = useState(computeBoundingBox(INITIAL_VIEW_STATE))

  let layers = []

  if (false)
  layers.push( new ScatterplotLayer({
    id: 'airbnb+houses-within-bbox',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/all-airbnb.json',
    pickable: true,
    opacity: 1.,
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 1,
    lineWidthMinPixels: 1,
    getPosition: d => [d[1], d[0]],
    onHover:() => {},
    onMouseOut: () => {},
    onClick: ({object}) => setOpenPopover(object[2]),
    getRadius: 1,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[0]))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  layers.push( new ScreenGridLayer({
    id: 'airbnb+houses-within-bbox',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/all-airbnb.json',
    opacity: 1.,
    getPosition: d => [d[1], d[0]],
    getWeight: () => 1,
    cellSizePixels: 30 - 10 * currentViewState.zoom,  //change on zoomLevel
    //colorRange,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[0]))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  if (false)
  layers.push( new ScatterplotLayer({
    id: 'scatterplot-layer',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/city_location.json',
    pickable: false,
    opacity: 1.,
    stroked: false,
    filled: true,
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 1,
    lineWidthMinPixels: 1,

    getPosition: d => {
      return d.reverse()
    },
    //getPosition: d => [d[0], d[1], 0],
    onClick: ({object}) => {
      let url = `https://www.airbnb.com/rooms/${object[0]}`
      window.open(url)
    },
    //getPosition: d => centroid,
    getRadius: (d, datum) => datum.index,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[1] / 365))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  let iconLayer = new IconLayer({
      id: 'icon-layer',
      //data: markers,
      data: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/bart-stations.json',
      pickable: true,
      // iconAtlas and iconMapping are required
      // getIcon: return a string
      iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
      iconMapping: ICON_MAPPING,
      getIcon: d => 'marker',
      onClick: (_)=> _.object.url,
      sizeScale: 150,
      getPosition: d => {
        return d.coordinates
        return [+ d.lon, + d.lat]
      },
      getSize: d => 50,
      getColor: d => { 
        console.log('wtf man')
        return [Math.random() * 255, 140, 0]
      }
    })
  //layers.push(iconLayer)

  const allCoordinates = []
  routes.forEach((route) => {
    route.geometry.coordinates.forEach((list, i ) => list.push(i / route.geometry.coordinates.length))
    allCoordinates.push(...route.geometry.coordinates)
  })

  let roads = new ScatterplotLayer({
    id: 'roads',
    data: allCoordinates,
    pickable: false,
    opacity: 1.,
    stroked: false,
    filled: true,
    radiusScale: 1,
    radiusMinPixels: 5,
    radiusMaxPixels: 5,
    lineWidthMinPixels: 1,
    getPosition: (one, two, three) => {
      return one
    },
    //getPosition: d => [d[0], d[1], 0],
    //getPosition: d => centroid,
    getRadius: d => 10,
    getFillColor: (_, datum) => {
      let rgb = d3.rgb(interpolateRainbow(_[2]))
      return [rgb.r, rgb.g, rgb.b]
      return [0, 255 * _[2], 255]
    },
    // getFilterValue: f => {
    //   //console.log(f)
    //   //f.properties.timeOfDay
    //   return Math.random () > .5
    // },  // in seconds
    //filterRange: [43200, 46800],  // 12:00 - 13:00
    //extensions: [new DataFilterExtension({filterSize: 1})]

  })
  //layers.push(roads)
  
  const fetchRoutes = async () => {
    //everytime you move, draw contiguous paths from center to 4 corners
    let {top, left, right, bottom} = currentViewState;
    let url = `${baseName}/osm_bbox/?min_lat=${top}&min_lng=${left}&max_lat=${bottom}&max_lng=${right}`;

    const response = await fetch(url);
    const json = await response.json();

    markers.push.apply(markers, json.places)

    setMarkers(markers)
    for (let i = 0; i < 1; i++) {
      let min_lat = left + .05 * Math.random(), 
      min_lng = top + .05 * Math.random(), 
      max_lat = right + .05 * Math.random(), 
      max_lng = bottom + .05 * Math.random();
      const req = await fetch(
        `https://api.mapbox.com/directions/v5/mapbox/driving/${min_lat},${min_lng};${max_lat},${max_lng}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg`
      );
      let json = await req.json();
      routes.push.apply(routes, json.routes)
      setRoutes(routes)
    }
    setPercent(0)
    setTimeout(function recur () {
      //setPercent(getPercent+.1)
      //if (getPercent < 1) setTimeout(recur, 100)
    }, 100)
    //setMarkers(json.places)
    //console.log(json.routes.length)
  }

  useEffect(() => {
    fetchRoutes();
  }, [currentViewState.left]);


  return (<>
    <h3 className="">World Map! - Scroll to zoom in to see every home in the world at a higher resolution + <button className="text-black">Click here to place a house and generate program instructions for robot to build it.</button></h3>
    <div className="relative" style={{left: `${props.left}px`, height: '600px'}}>
    <PopOver open={openPopover} setOpen={setOpenPopover}/>
    <ColorLabels></ColorLabels>
    <ProgressBar percentage={100}></ProgressBar>
    <DeckGL
        width={1200}
        height={600}
      layers={layers}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
      onWebGLInitialized={onInitialized}
      onViewStateChange={_.debounce(({viewState}) => {
        setViewState(computeBoundingBox(viewState))
      }, 1000)}
    >
      <Map 
       glOptions={{preserveDrawingBuffer: true}}
      reuseMaps mapLib={maplibregl} mapStyle={MAP_STYLE} preventStyleDiffing={true}>
      </Map>
    </DeckGL>

    </div>
    <div className="grid grid-cols-2">
      <div className="">
        <div >
          <div>Latitude: {currentViewState.bottom.toPrecision(4)}  |   {currentViewState.top.toPrecision(4)}</div>
          <div>Longitude: {currentViewState.left.toPrecision(4)} |  {currentViewState.right.toPrecision(4)}</div>
          <div>Total Roads Drawn: {routes.length} / 1 billion</div>
          <div>Places Of Interest Drawn: {markers.length}</div>
          <div>Airbnbs rendered on Screen: {routes.length} /  7 million</div>
          <div>Benches rendered on Screen: {markers.length} /  7 million</div>
          <div>Houses rendered on Screen:  / 135 million from  
            <span className="text-blue-500"> Zillow</span> 
            <span className="text-red-500">Redfin</span> 
            <span className="text-green-500">Compass</span> 
            <span className="text-yellow-500 hidden">immobilienscout24.de</span>
          </div>
          </div>
        </div>
        <span className="flex">
          <DonutChart random={Math.random()}/>
          <Histogram random={Math.random()} />
        </span>
    </div>
    </>
  );
}

function computeBoundingBox(viewPort) {
  const viewport = new WebMercatorViewport({...viewPort});
  const topLeft = viewport.unproject([0, 0]);
  const bottomRight = viewport.unproject([viewport.width, viewport.height]);

  const boundingBox = {
    top: clamp(topLeft[1], -90, 90),
    left: clamp(topLeft[0], -180, 180),
    bottom: clamp(bottomRight[1], -90, 90),
    right: clamp(bottomRight[0], -180, 180),
    centroid: [viewport.latitude, viewport.longitude],
    zoom: viewPort.zoom
  };
  console.log(viewPort)
  return boundingBox
}

export default function AirbnbPriceMap (){
const [isShowing, setIsShowing] = useState(true)
  return ( <>
        {/* <Transition
        show={isShowing}
        enter="transition-opacity duration-75"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-150"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      > */}
     <AirbnbWorldMap />
    {/* </Transition> */}
  </>)
}
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

function getTooltip(params) {
    if (! params.picked) return 
    return JSON.stringify(params.object, null, 2)
}

function ColorLabels() {
    return <></>
}
function clamp(_, min, max) {
  return Math.max(Math.min(_, max), min)
}
function ProgressBar (props) {
  let style = {width: `${innerWidth}px`}
  return (<div className="fixed left-0 top-0 progress-bar h-4 w-full" style={style}></div>)
}

function onInitialized (gl) {
  if (!isWebGL2(gl)) {
    console.warn('GPU aggregation is not supported'); // eslint-disable-line
    if (disableGPUAggregation) {
      disableGPUAggregation();
    }
  }
};