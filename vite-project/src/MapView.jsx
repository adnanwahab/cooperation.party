import React from 'react';
import {createRoot} from 'react-dom/client';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {AmbientLight, PointLight, LightingEffect} from '@deck.gl/core';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';
import ReactMap, {Source, Layer, Marker} from 'react-map-gl';

import {GeoJsonLayer} from '@deck.gl/layers';
import {IconLayer} from '@deck.gl/layers';
//import {interpolatePurples} from  'https://cdn.jsdelivr.net/npm/d3-scale-chromatic@3'
import {interpolatePurples} from "https://cdn.skypack.dev/d3-scale-chromatic@3";

import {cellToLatLng} from 'h3-js'
const ICON_MAPPING = {
    marker: {x: 0, y: 0, width: 128, height: 128, mask: true}
  };
  

import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';
const COLOR_SCALE = [
    [65, 182, 196],
    [127, 205, 187],
    [199, 233, 180],
    [237, 248, 177],
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [189, 0, 38],
    [128, 0, 38],
]

const colorScale = (x) => {
    const i = Math.round(x * 7) + 4
    if (x < 0) {
        return COLOR_SCALE[i] || COLOR_SCALE[0]
    }
    return COLOR_SCALE[i] || COLOR_SCALE[COLOR_SCALE.length - 1]
}
// Source data CSV
const DATA_URL =
  'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv'; // eslint-disable-line

const ambientLight = new AmbientLight({
  color: [255, 255, 255],
  intensity: 1.0
});

const pointLight1 = new PointLight({
  color: [255, 255, 255],
  intensity: 0.8,
  position: [-0.144528, 49.739968, 80000]
});

const pointLight2 = new PointLight({
  color: [255, 255, 255],
  intensity: 0.8,
  position: [-3.807751, 54.104682, 8000]
});

const lightingEffect = new LightingEffect({ambientLight, pointLight1, pointLight2});

const material = {
  ambient: 0.64,
  diffuse: 0.6,
shininess: 32,
  specularColor: [51, 51, 51]
};

//139.65,34.99038


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
// const getCsv = await d3.csv(DATA_URL, (data) => {
//     return [Number(data.lng), Number(data.lat)]
// })

// console.log(getCsv)
//(error, response) => { return response.map(d => ) }
/* eslint-disable react/no-deprecated */




export default function App({
    // find centroid -> draw isochrone and ensure that all people are within 20 min train
    h3_hexes,
    routes={},
    centroid=[0,0],
reports,
    hexes,
    _houses,
    data,
  mapStyle = MAP_STYLE,
  radius = 1000,
  upperPercentile = 100,
  coverage = 1,
  left
}) {
  if (routes.length) { 
    //routes = routes[0].routes[0].geometry
    routes = routes.map(_ => _.routes[0].geometry)
    //console.log(routes, 'geometry')
    centroid = routes[0].coordinates[0]
    routes = routes
  }
else {
  // return (<>
  // This is a map with a clickable legend 
  // </>
  // )
}
  let layers = [
    // new IconLayer({
    //     id: 'icon-layer',
    //     data: data,
    //     pickable: true,
    //     // iconAtlas and iconMapping are required
    //     // getIcon: return a string
    //     iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
    //     iconMapping: ICON_MAPPING,
    //     getIcon: d => 'marker',
    //     onClick: (_)=> _.object.url,
    //     sizeScale: 15,
    //     getPosition: d => [+ d.longitude, + d.latitude],
    //     getSize: d => 5,
    //     getColor: d => [Math.random() * 255, 140, 0]
    //   }),
    // new H3HexagonLayer({
    //   id: 'h3',
    //   getFillColor: _ => Object.values(d3.rgb(d3.interpolatePurples(_[1].vending_machine / 100))).slice(0, 3),
    //   data: Object.entries(hexes),
    //   elevationRange: [0, 0],
    //   elevationScale: 1,
    //   getHexagon: d => d[0],
    //   pickable: true,
    //   radius,
    //   onClick: (_) => console.log(_),
    //   //upperPercentile,
    //   material,
    //   //opacity:.9,
    //   extruded: false,
    // //   transitions: {
    // //     elevationScale: 3000
    // //   }
    // }),
    // new GeoJsonLayer({
    //     id: 'geojson-layer',
    //     data: isochrone.features[0],
    // // data: isochrone,
    //     pickable: false,
    //     //lineWidthScale: 20,
    //     lineWidthMinPixels: 2,
    //     getFillColor: [160, 160, 180, 200],
    //     // getPointRadius: 100,
    //     getLineWidth: 10,
    //     opacity: .3,
    //     getFillColor: (f) => colorScale(Math.random()),
    //     getElevation: -1
    // }),

   
  ];

  routes.map((route, route_index) => {
    centroid = route.coordinates[0]
    console.log('centroid - ', centroid)
    let newLayer = new GeoJsonLayer({
      id: 'geojson',
      //data: 
      data: route,
      stroked: false,
      filled: false,
      lineWidthMinPixels: 0.5,
      parameters: {
        depthTest: false
      },

      getLineColor: function (_) {
          //console.log(_)
          return [255, 0, 0, 255]
      },
      getLineWidth: function () {
          return 20
      },

      pickable: true,
     // onHover: setHoverInfo,

      // updateTriggers: {
      //   getLineColor: {year},
      //   getLineWidth: {year}
      // },

      transitions: {
        getLineColor: 1000,
        getLineWidth: 1000
      }
    })
    layers.push(newLayer)
  })

  h3_hexes
  centroid = [0,0]
  if (h3_hexes) {
    let len = Object.keys(h3_hexes).length

    
    let places  = Object.keys(h3_hexes).map(_ => {
      return cellToLatLng(_)
    }).forEach(_ => { 
      centroid[0] += _[0]
      centroid[1] += _[1]
    })
    
    console.log('i am a centroid', centroid)
    centroid[0] /= len
    centroid[1] /= len
    console.log('i am a centroid', centroid)

    layers = [
      new H3HexagonLayer({
        id: 'h3',
        getFillColor: _ => [255* Math.random(), 255, 255, 255],
        //Object.values(d3.rgb(d3.interpolatePurples(_[1].vending_machine / 100))).slice(0, 3),
        data: Object.entries(h3_hexes),
        elevationRange: [0, 0],
        elevationScale: 1,
        getHexagon: d => d[0],
        pickable: true,
        radius,
        onClick: (_) => console.log(_),
        //upperPercentile,
        material,
        //opacity:.9,
        extruded: false,
      //   transitions: {
      //     elevationScale: 3000
      //   }
      }),
    ]
  }
  console.log('i am a centroid', centroid)




const INITIAL_VIEW_STATE = {
  longitude: + centroid[1],
  latitude: + centroid[0],
  //   longitude: 18,
  // latitude: -33,
  zoom: 10,
  maxZoom: 20,
  pitch: 0,
  bearing: 0
}



  return (
<div className=" absolute h-96" style={{left: `${left}px`}}>
    <h3>Colorize Hexes by suitability for given schedule.</h3>
    <Legend></Legend>
    <DeckGL
        width={500}
        height={500}

      layers={layers}
      effects={[lightingEffect]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
    >
      <Map 
       glOptions={{preserveDrawingBuffer: true}}
      reuseMaps mapLib={maplibregl} mapStyle={mapStyle} preventStyleDiffing={true}>
      </Map>
    </DeckGL>
    </div>
  );
}

//hex colors
function Legend() {
    return <></>
}