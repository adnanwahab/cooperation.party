import React, { useEffect } from 'react';
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


import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries

//import citiesList from "../../city_locations.json";

const city_locations = [
  "Ghent--Flemish-Region--Belgium",
  "Nashville--Tennessee--United-States",
  "Menorca--Islas-Baleares--Spain",
  "Melbourne--Victoria--Australia",
  "Broward-County--Florida--United-States",
  "Pays-Basque--Pyrénées-Atlantiques--France",
  "Bangkok--Central-Thailand--Thailand",
  "Bordeaux--Nouvelle-Aquitaine--France",
  "Mornington-Peninsula--Victoria--Australia",
  "Boston--Massachusetts--United-States",
  "New-Zealand",
  "Asheville--North-Carolina--United-States",
  "Newark--New-Jersey--United-States",
  "Malta",
  "Mexico-City--Distrito-Federal--Mexico",
  "Puglia--Puglia--Italy",
  "Lyon--Auvergne-Rhone-Alpes--France",
  "Quebec-City--Quebec--Canada",
  "Western-Australia--Western-Australia--Australia",
  "Buenos-Aires--Ciudad-Autónoma-de-Buenos-Aires--Argentina",
  "Bergamo--Lombardia--Italy",
  "Vancouver--British-Columbia--Canada",
  "Porto--Norte--Portugal",
  "Rio-de-Janeiro--Rio-de-Janeiro--Brazil",
  "Portland--Oregon--United-States",
  "Lisbon--Lisbon--Portugal",
  "Twin-Cities-MSA--Minnesota--United-States",
  "Cambridge--Massachusetts--United-States",
  "Columbus--Ohio--United-States",
  "Valencia--Valencia--Spain",
  "Santiago--Región-Metropolitana-de-Santiago--Chile",
  "Sicily--Sicilia--Italy",
  "Edinburgh--Scotland--United-Kingdom",
  "Belize--Belize--Belize",
  "Jersey-City--New-Jersey--United-States",
  "Thessaloniki--Central-Macedonia--Greece",
  "Toronto--Ontario--Canada",
  "Antwerp--Flemish-Region--Belgium",
  "Cape-Town--Western-Cape--South-Africa",
  "Vienna--Vienna--Austria",
  "Hawaii--Hawaii--United-States",
  "Tokyo--Japan",
  "Santa-Cruz-County--California--United-States",
  "Pacific-Grove--California--United-States",
  "Clark-County--NV--Nevada--United-States",
  "Denver--Colorado--United-States",
  "New-Orleans--Louisiana--United-States",
  "New-York-City--New-York--United-States",
  "Barwon-South-West--Vic--Victoria--Australia",
  "Paris--Île-de-France--France",
  "Hong-Kong--Hong-Kong--China",
  "Bristol--England--United-Kingdom",
  "Athens--Attica--Greece",
  "Geneva--Geneva--Switzerland",
  "Brussels--Brussels--Belgium",
  "Washington--D.C.--District-of-Columbia--United-States",
  "Istanbul--Marmara--Turkey",
  "Dublin--Leinster--Ireland",
  "Vaud--Vaud--Switzerland",
  "Taipei--Northern-Taiwan--Taiwan",
  "Euskadi--Euskadi--Spain",
  "Winnipeg--Manitoba--Canada",
  "Ireland",
  "Prague--Prague--Czech-Republic",
  "Zurich--Zürich--Switzerland",
  "Montreal--Quebec--Canada",
  "Singapore--Singapore--Singapore",
  "Venice--Veneto--Italy",
  "Greater-Manchester--England--United-Kingdom",
  "New-Brunswick--New-Brunswick--Canada",
  "Northern-Rivers--New-South-Wales--Australia",
  "Mid-North-Coast--New-South-Wales--Australia",
  "San-Mateo-County--California--United-States",
  "Chicago--Illinois--United-States",
  "Los-Angeles--California--United-States",
  "Stockholm--Stockholms-län--Sweden",
  "Santa-Clara-County--California--United-States",
  "Sevilla--Andalucía--Spain",
  "San-Diego--California--United-States",
//  "Beijing--Beijing--China",
  "Salem--OR--Oregon--United-States",
  "San-Francisco--California--United-States",
  "Rotterdam--South-Holland--The-Netherlands",
  "Victoria--British-Columbia--Canada",
  "Rhode-Island--Rhode-Island--United-States",
  "Barossa-Valley--South-Australia--Australia",
  "Oslo--Oslo--Norway",
  "Fort-Worth--Texas--United-States",
  "Munich--Bavaria--Germany",
  "Sydney--New-South-Wales--Australia",
//  "Shanghai--Shanghai--China",
  "Austin--Texas--United-States",
  "Tasmania--Tasmania--Australia",
  "Barcelona--Catalonia--Spain",
  "Mallorca--Islas-Baleares--Spain",
  "Berlin--Berlin--Germany",
  "Riga--Riga--Latvia",
  "Bozeman--Montana--United-States",
  "Oakland--California--United-States",
  "Copenhagen--Hovedstaden--Denmark",
  "Rome--Lazio--Italy",
  "South-Aegean--South-Aegean--Greece",
  "Crete--Crete--Greece",
  "Bologna--Emilia-Romagna--Italy",
  "Milan--Lombardy--Italy",
  "The-Hague--South-Holland--The-Netherlands",
  "Florence--Toscana--Italy",
  "Malaga--Andalucía--Spain",
  "Girona--Catalonia--Spain",
  "Amsterdam--North-Holland--The-Netherlands",
  "Dallas--Texas--United-States",
  "Seattle--Washington--United-States",
  "Madrid--Comunidad-de-Madrid--Spain",
  "Naples--Campania--Italy",
  "London--England--United-Kingdom"
]


function AirbnbWorldMap(props) {

  const layers = []
  
  const INITIAL_VIEW_STATE = {
    longitude: 0,
    latitude: 0,
    //   longitude: 18,
    // latitude: -33,
    zoom: 0,
    maxZoom: 20,
    pitch: 0,
    bearing: 0
  }    
  return (<>
    <h3 class="">World Map</h3>
    <div className=" relative h-96" style={{left: `${props.left}px`}}>
    <Legend></Legend>
    <DeckGL
        width={750}
        height={1000}
      layers={layers}
      //effects={[lightingEffect]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
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

export default function AirbnbPriceMap (props){
  const data = []
  const [cityData, setCityData] = useState([]);

  //figure promise.all http streaming 
  //add a progress bar - few seconds to render -> rainbow

  // city_locations.slice(0, 10).forEach(async (city_name) => {
  //   const req = await fetch(`https://shelbernstein.ngrok.io/data/airbnb/apt/${city_name}`);
  //   const json = await req.json()
  //   for (let key in json) {
  //     data.push(
  //       json[key]
  //     )
  //   }
  // })
  // Promise.all()

  useEffect(() => {

  }, [])


//  const otherMaps = Object.entries(props.data).map((pair) => {
//     return <JustMap title={pair[0]} data={pair[1]} left={0} />
//   })
const otherMaps = null

  return ( <>
    <AirbnbWorldMap data={data}></AirbnbWorldMap>
  
    {otherMaps}
  </>)
}
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

function computeCentroid (list) {
    let centroid = [0, 0]

    list.map(_ => _.slice(0,2).map(_ => parseFloat(_)))
    .forEach(pair => {
        centroid[0] += pair[0]
        centroid[1] += pair[1]
    })

    centroid[0] /= list.length
    centroid[1] /= list.length


    return centroid
}


function JustMap(props) {
    const centroid = computeCentroid(props.data)
    console.log('onlyMap',props.data)
    const layer = new ScatterplotLayer({
        id: 'scatterplot-layer',
        data:props.data,
        pickable: true,
        opacity: 0.8,
        stroked: true,
        filled: true,
        radiusScale: 6,
        radiusMinPixels: 1,
        radiusMaxPixels: 100,
        lineWidthMinPixels: 1,
        getPosition: d => d.slice(0,2).reverse(),
        //getPosition: d => centroid,
        getRadius: d => 30,
        getFillColor: d => [(d[2]) * 255, 0, 255],
        getLineColor: d => [0, 0, 0]
      });
      const layers = [layer]

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

    return (<>
    <h3 class="">{props.title}</h3>
    <div className=" relative h-96" style={{left: `${props.left}px`}}>
    <Legend></Legend>
    <DeckGL
        width={400}
        height={300}
      layers={layers}
      //effects={[lightingEffect]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
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




function App({
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
    //centroid = routes[0].coordinates[0]
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
    //centroid = route.coordinates[0]
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

  if (h3_hexes) {

   
  

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






  return (
<div className=" absolute h-96" style={{left: `${left}px`}}>
    <h3>Colorize Hexes by suitability for given schedule.</h3>
    <Legend></Legend>
    <DeckGL
        width={400}
        height={400}

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