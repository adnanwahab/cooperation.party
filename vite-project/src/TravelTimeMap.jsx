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

const ICON_MAPPING = {
    marker: {x: 0, y: 0, width: 128, height: 128, mask: true}
  };
  
  const List = (list) => 
  (<ul className="overflow-scroll h-64">
    <li key="item font-white">{list.length}</li>
    {list.map((item, idx) => <li key={idx}>{item}</li>)}
  </ul>)

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
data,
  mapStyle = MAP_STYLE,
  radius = 1000,
  upperPercentile = 100,
  coverage = 1
}) {
  const layers = [
    // new IconLayer({
    //     id: 'icon-layer',
    //     data: data.map(_ => Object.values(_._houses)).flat(),
    //     pickable: true,
    //     // iconAtlas and iconMapping are required
    //     // getIcon: return a string
    //     iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
    //     iconMapping: ICON_MAPPING,
    //     getIcon: d => 'marker',
    //     onClick: (_)=> _.object.url,
    //     sizeScale: 15,
    //     getPosition: d => d.map(parseFloat),
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
    //data[0][0].isochrone.features[0]
    new GeoJsonLayer({
        id: 'geojson-layer',
    //    data: data.map(_=> _.isochrone.features[0]),
   data: data[0].isochrone.features[0],
        pickable: false,
        //lineWidthScale: 20,
        lineWidthMinPixels: 2,
        getFillColor: [160, 160, 180, 200],
        // getPointRadius: 100,
        getLineWidth: 10,
        opacity: .3,
        getFillColor: (f) => colorScale(Math.random()),
        getElevation: -1
    }),

   
  ];

  const renderMarker = (point)=> {
    let [lat, lng] = point
   return (<Marker
        key={`marker-${Math.random()}`}
        longitude={lng}
        latitude={lat}
        anchor="bottom"
        onClick={e => {
            // If we let the click event propagates to the map, it will immediately close the popup
            // with `closeOnClick: true`
            e.originalEvent.stopPropagation();
            // setPopupInfo(city);
        }}
        >
        {/* <Pin /> */}
        </Marker>)
  }
//debugger
  let centroid = data[0].centroid
const INITIAL_VIEW_STATE = {
  longitude: parseFloat(centroid.longitude),
  latitude: parseFloat(centroid.latitude),
  zoom: 11,
  maxZoom: 20,
  pitch: 0,
  bearing: 0
}

  return (
<div className="relative h-96 mt-50">
    <h3>This is a barchart of all airbnbs in that city</h3>
    <h3>This is a map of all airbnbs within commute distance to the places that you specified in the document</h3>
<div class="relative mt-4">
    <DeckGL
        width={500}
        height={500}

      layers={layers}
      effects={[lightingEffect]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      
    >
      <Map reuseMaps mapLib={maplibregl} mapStyle={mapStyle} preventStyleDiffing={true}>
{/* {markers} */}
      </Map>
    </DeckGL>
    {/* <div class="absolute right-0"> {List(Object.keys(_houses).map(_ => {
        return <a href={'https://airbnb.com/rooms/'+_}>{'https://airbnb.com/rooms/'+_}</a>
    }))}</div> */}
    </div>
    </div>
  );
}