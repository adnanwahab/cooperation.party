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
const INITIAL_VIEW_STATE = {
    longitude: 139.6503,
    latitude: 35.6762,
    zoom: 11,
    maxZoom: 20,
    pitch: 0,
    bearing: 0
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




    return params.object.url
}
// const getCsv = await d3.csv(DATA_URL, (data) => {
//     return [Number(data.lng), Number(data.lat)]
// })

// console.log(getCsv)
//(error, response) => { return response.map(d => ) }
/* eslint-disable react/no-deprecated */

// let benches = await d3.json('https://pypypy.ngrok.io/data/airbnb/h3_poi/bench.json')
// console.log(benches)


export default function App({
    // find centroid -> draw isochrone and ensure that all people are within 20 min train
reports,
    isochrone,
    houses,
  mapStyle = MAP_STYLE,
  radius = 1000,
  upperPercentile = 100,
  coverage = 1
}) {
//const data = getCsv
console.log(reports)
  const layers = [
    new IconLayer({
        id: 'icon-layer',
        data: houses,
        pickable: true,
        // iconAtlas and iconMapping are required
        // getIcon: return a string
        iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
        iconMapping: ICON_MAPPING,
        getIcon: d => 'marker',
        onClick: (_)=> _.object.url,
        sizeScale: 15,
        getPosition: d => d.location.reverse().concat(100),
        getSize: d => 5,
        getColor: d => [Math.random() * 255, 140, 0]
      }),
    new H3HexagonLayer({
      id: 'heatmap',
      colorRange,
      coverage,
      getFillColor: d => [255, (Math.random * 500) * 255, 0],
      data: Object.entries(reports),
      elevationRange: [0, 0],
      elevationScale: 1,
      getHexagon: d => d[0],
      pickable: true,
      radius,
      upperPercentile,
      material,
      opacity:.1,
    //   transitions: {
    //     elevationScale: 3000
    //   }
    }),
    new GeoJsonLayer({
        id: 'geojson-layer',
        data: isochrone.features[0],
    // data: isochrone,
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
let        markers = houses.map(_ => {
    return renderMarker(_['location'])
})

  return (
<div>
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
    </div>
  );
}
