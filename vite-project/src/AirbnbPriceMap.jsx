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


import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries

//import citiesList from "../../city_locations.json";

const city_locations = {
  "Ghent--Flemish-Region--Belgium": {
    "longitude": "3.717424",
    "latitude": "51.054342"
  },
  "Nashville--Tennessee--United-States": {
    "longitude": "-86.781602",
    "latitude": "36.162664"
  },
  "Menorca--Islas-Baleares--Spain": {
    "longitude": "3.859550",
    "latitude": "39.998211"
  },
  "Melbourne--Victoria--Australia": {
    "longitude": "144.963058",
    "latitude": "-37.813628"
  },
  "Broward-County--Florida--United-States": {
    "longitude": "-80.357389",
    "latitude": "26.138279"
  },
  "Pays-Basque--Pyrénées-Atlantiques--France": {
    "longitude": "-1.475207",
    "latitude": "43.184289"
  },
  "Bangkok--Central-Thailand--Thailand": {
    "longitude": "100.516667",
    "latitude": "13.75"
  },
  "Bordeaux--Nouvelle-Aquitaine--France": {
    "longitude": "-0.57918",
    "latitude": "44.837789"
  },
  "Mornington-Peninsula--Victoria--Australia": {
    "longitude": "145.049576",
    "latitude": "-38.285405"
  },
  "Boston--Massachusetts--United-States": {
    "longitude": "-71.058880",
    "latitude": "42.360082"
  },
  "New-Zealand": {
    "longitude": "174.885971",
    "latitude": "-40.900557"
  },
  "Asheville--North-Carolina--United-States": {
    "longitude": "-82.554016",
    "latitude": "35.595058"
  },
  "Newark--New-Jersey--United-States": {
    "longitude": "-74.172366",
    "latitude": "40.735657"
  },
  "Malta": {
    "longitude": "14.375416",
    "latitude": "35.937496"
  },
  "Mexico-City--Distrito-Federal--Mexico": {
    "longitude": "-99.133209",
    "latitude": "19.432608"
  },
  "Puglia--Puglia--Italy": {
    "longitude": "16.863909",
    "latitude": "40.792839"
  },
  "Lyon--Auvergne-Rhone-Alpes--France": {
    "longitude": "4.835659",
    "latitude": "45.759362"
  },
  "Quebec-City--Quebec--Canada": {
    "longitude": "-71.208180",
    "latitude": "46.813878"
  },
  "Western-Australia--Western-Australia--Australia": {
    "longitude": "121.887406",
    "latitude": "-31.950527"
  },
  "Buenos-Aires--Ciudad-Autónoma-de-Buenos-Aires--Argentina": {
    "longitude": "-58.417309",
    "latitude": "-34.611996"
  },
  "Bergamo--Lombardia--Italy": {
    "longitude": "9.692345",
    "latitude": "45.694454"
  },
  "Vancouver--British-Columbia--Canada": {
    "longitude": "-123.121644",
    "latitude": "49.282729"
  },
  "Porto--Norte--Portugal": {
    "longitude": "-8.611003",
    "latitude": "41.157944"
  },
  "Rio-de-Janeiro--Rio-de-Janeiro--Brazil": {
    "longitude": "-43.172896",
    "latitude": "-22.906846"
  },
  "Portland--Oregon--United-States": {
    "longitude": "-122.673647",
    "latitude": "45.505106"
  },
  "Lisbon--Lisbon--Portugal": {
    "longitude": "-9.139337",
    "latitude": "38.722252"
  },
  "Twin-Cities-MSA--Minnesota--United-States": {
    "longitude": "-93.265011",
    "latitude": "44.977753"
  },
  "Cambridge--Massachusetts--United-States": {
    "longitude": "-71.109733",
    "latitude": "42.373611"
  },
  "Columbus--Ohio--United-States": {
    "longitude": "-82.998794",
    "latitude": "39.961176"
  },
  "Valencia--Valencia--Spain": {
    "longitude": "-0.375951",
    "latitude": "39.469907"
  },
  "Santiago--Región-Metropolitana-de-Santiago--Chile": {
    "longitude": "-70.6483",
    "latitude": "-33.4489"
  },
  "Sicily--Sicilia--Italy": {
    "longitude": "14.532899",
    "latitude": "37.599994"
  },
  "Edinburgh--Scotland--United-Kingdom": {
    "longitude": "-3.188267",
    "latitude": "55.953251"
  },
  "Belize--Belize--Belize": {
    "longitude": "-88.760092",
    "latitude": "17.497713"
  },
  "Jersey-City--New-Jersey--United-States": {
    "longitude": "-74.077636",
    "latitude": "40.728157"
  },
  "Thessaloniki--Central-Macedonia--Greece": {
    "longitude": "22.942199",
    "latitude": "40.640063"
  },
  "Toronto--Ontario--Canada": {
    "longitude": "-79.383184",
    "latitude": "43.653225"
  },
  "Antwerp--Flemish-Region--Belgium": {
    "longitude": "4.402464",
    "latitude": "51.219447"
  },
  "Cape-Town--Western-Cape--South-Africa": {
    "longitude": "18.424055",
    "latitude": "-33.924869"
  },
  "Vienna--Vienna--Austria": {
    "longitude": "16.373819",
    "latitude": "48.208174"
  },
  "Hawaii--Hawaii--United-States": {
    "longitude": "-155.665857",
    "latitude": "20.796697"
  },
  "Tokyo--Japan": {
    "longitude": "139.691706",
    "latitude": "35.689487"
  },
  "Santa-Cruz-County--California--United-States": {
    "longitude": "-122.030799",
    "latitude": "37.020001"
  },
  "Pacific-Grove--California--United-States": {
    "longitude": "-121.916622",
    "latitude": "36.617737"
  },
  "Clark-County--NV--Nevada--United-States": {
    "longitude": "-115.159272",
    "latitude": "36.214496"
  },
  "Denver--Colorado--United-States": {
    "longitude": "-104.990251",
    "latitude": "39.739236"
  },
  "New-Orleans--Louisiana--United-States": {
    "longitude": "-90.071532",
    "latitude": "29.951066"
  },
  "New-York-City--New-York--United-States": {
    "longitude": "-74.006015",
    "latitude": "40.712776"
  },
  "Barwon-South-West--Vic--Victoria--Australia": {
    "longitude": "143.833133",
    "latitude": "-38.261111"
  },
  "Paris--Île-de-France--France": {
    "longitude": "2.352222",
    "latitude": "48.856614"
  },
  "Hong-Kong--Hong-Kong--China": {
    "longitude": "114.169361",
    "latitude": "22.396428"
  },
  "Bristol--England--United-Kingdom": {
    "longitude": "-2.587910",
    "latitude": "51.454514"
  },
  "Athens--Attica--Greece": {
    "longitude": "23.727539",
    "latitude": "37.983810"
  },
  "Geneva--Geneva--Switzerland": {
    "longitude": "6.143158",
    "latitude": "46.204394"
  },
  "Brussels--Brussels--Belgium": {
    "longitude": "4.351710",
    "latitude": "50.850340"
  },
  "Washington--D.C.--District-of-Columbia--United-States": {
    "longitude": "-77.036871",
    "latitude": "38.895110"
  },
  "Istanbul--Marmara--Turkey": {
    "longitude": "28.978359",
    "latitude": "41.008238"
  },
  "Dublin--Leinster--Ireland": {
    "longitude": "-6.26031",
    "latitude": "53.349805"
  },
  "Vaud--Vaud--Switzerland": {
    "longitude": "6.632273",
    "latitude": "46.603354"
  },
  "Taipei--Northern-Taiwan--Taiwan": {
    "longitude": "121.565426",
    "latitude": "25.032969"
  },
  "Euskadi--Euskadi--Spain": {
    "longitude": "-2.988556",
    "latitude": "43.263012"
  },
  "Winnipeg--Manitoba--Canada": {
    "longitude": "-97.137494",
    "latitude": "49.895138"
  },
  "Ireland": {
    "longitude": "-7.692054",
    "latitude": "53.142367"
  },
  "Prague--Prague--Czech-Republic": {
    "longitude": "14.437800",
    "latitude": "50.075538"
  },
  "Zurich--Zürich--Switzerland": {
    "longitude": "8.541694",
    "latitude": "47.376886"
  },
  "Montreal--Quebec--Canada": {
    "longitude": "-73.567256",
    "latitude": "45.501690"
  },
  "Singapore--Singapore--Singapore": {
    "longitude": "103.819836",
    "latitude": "1.352083"
  },
  "Venice--Veneto--Italy": {
    "longitude": "12.315515",
    "latitude": "45.440847"
  },
  "Greater-Manchester--England--United-Kingdom": {
    "longitude": "-2.242631",
    "latitude": "53.483959"
  },
  "New-Brunswick--New-Brunswick--Canada": {
    "longitude": "-66.461914",
    "latitude": "46.498390"
  },
  "Northern-Rivers--New-South-Wales--Australia": {
    "longitude": "153.336990",
    "latitude": "-29.458490"
  },
  "Mid-North-Coast--New-South-Wales--Australia": {
    "longitude": "152.846792",
    "latitude": "-30.674364"
  },
  "San-Mateo-County--California--United-States": {
    "longitude": "-122.355546",
    "latitude": "37.554512"
  },
  "Chicago--Illinois--United-States": {
    "longitude": "-87.629799",
    "latitude": "41.878113"
  },
  "Los-Angeles--California--United-States": {
    "longitude": "-118.243683",
    "latitude": "34.052235"
  },
  "Stockholm--Stockholms-län--Sweden": {
    "longitude": "18.068581",
    "latitude": "59.329323"
  },
  "Santa-Clara-County--California--United-States": {
    "longitude": "-121.955235",
    "latitude": "37.354108"
  },
  "Sevilla--Andalucía--Spain": {
    "longitude": "-5.984459",
    "latitude": "37.389092"
  },
  "San-Diego--California--United-States": {
    "longitude": "-117.161087",
    "latitude": "32.715736"
  },
  "Beijing--Beijing--China": {
    "longitude": "116.407396",
    "latitude": "39.904200"
  },
  "Salem--OR--Oregon--United-States": {
    "longitude": "-123.035096",
    "latitude": "44.942898"
  },
  "San-Francisco--California--United-States": {
    "longitude": "-122.419416",
    "latitude": "37.774929"
  },
  "Rotterdam--South-Holland--The-Netherlands": {
    "longitude": "4.477732",
    "latitude": "51.922500"
  },
  "Victoria--British-Columbia--Canada": {
    "longitude": "-123.365644",
    "latitude": "48.428421"
  },
  "Rhode-Island--Rhode-Island--United-States": {
    "longitude": "-71.412834",
    "latitude": "41.580095"
  },
  "Barossa-Valley--South-Australia--Australia": {
    "longitude": "138.985688",
    "latitude": "-34.533333"
  },
  "Oslo--Oslo--Norway": {
    "longitude": "10.752245",
    "latitude": "59.913869"
  },
  "Fort-Worth--Texas--United-States": {
    "longitude": "-97.330053",
    "latitude": "32.755488"
  },
  "Munich--Bavaria--Germany": {
    "longitude": "11.582017",
    "latitude": "48.135125"
  },
  "Sydney--New-South-Wales--Australia": {
    "longitude": "151.209295",
    "latitude": "-33.868820"
  },
  "Shanghai--Shanghai--China": {
    "longitude": "121.473701",
    "latitude": "31.230416"
  },
  "Austin--Texas--United-States": {
    "longitude": "-97.743060",
    "latitude": "30.267153"
  },
  "Tasmania--Tasmania--Australia": {
    "longitude": "146.609104",
    "latitude": "-41.454520"
  },
  "Barcelona--Catalonia--Spain": {
    "longitude": "2.173404",
    "latitude": "41.385064"
  },
  "Mallorca--Islas-Baleares--Spain": {
    "longitude": "2.650160",
    "latitude": "39.695263"
  },
  "Berlin--Berlin--Germany": {
    "longitude": "13.4050",
    "latitude": "52.5200"
  },
  "Riga--Riga--Latvia": {
    "longitude": "24.105186",
    "latitude": "56.949648"
  },
  "Bozeman--Montana--United-States": {
    "longitude": "-111.044718",
    "latitude": "45.677784"
  },
  "Oakland--California--United-States": {
    "longitude": "-122.271111",
    "latitude": "37.804363"
  },
  "Copenhagen--Hovedstaden--Denmark": {
    "longitude": "12.568337",
    "latitude": "55.676097"
  },
  "Rome--Lazio--Italy": {
    "longitude": "12.496366",
    "latitude": "41.902783"
  },
  "South-Aegean--South-Aegean--Greece": {
    "longitude": "24.997377",
    "latitude": "36.393156"
  },
  "Crete--Crete--Greece": {
    "longitude": "24.805272",
    "latitude": "35.240117"
  },
  "Bologna--Emilia-Romagna--Italy": {
    "longitude": "11.342616",
    "latitude": "44.494190"
  },
  "Milan--Lombardy--Italy": {
    "longitude": "9.190498",
    "latitude": "45.464203"
  },
  "The-Hague--South-Holland--The-Netherlands": {
    "longitude": "4.300700",
    "latitude": "52.070498"
  },
  "Florence--Toscana--Italy": {
    "longitude": "11.255814",
    "latitude": "43.769562"
  },
  "Malaga--Andalucía--Spain": {
    "longitude": "-4.421398",
    "latitude": "36.721273"
  },
  "Girona--Catalonia--Spain": {
    "longitude": "2.825989",
    "latitude": "41.979401"
  },
  "Amsterdam--North-Holland--The-Netherlands": {
    "longitude": "4.895168",
    "latitude": "52.370216"
  },
  "Dallas--Texas--United-States": {
    "longitude": "-96.796987",
    "latitude": "32.776664"
  },
  "Seattle--Washington--United-States": {
    "longitude": "-122.332071",
    "latitude": "47.606209"
  },
  "Madrid--Comunidad-de-Madrid--Spain": {
    "longitude": "-3.703791",
    "latitude": "40.416775"
  },
  "Naples--Campania--Italy": {
    "longitude": "14.268120",
    "latitude": "40.852167"
  },
  "London--England--United-Kingdom": {
    "longitude": "-0.128002",
    "latitude": "51.507351"
  }
}


function AirbnbWorldMap() {

  
  return (<>
    <h3 class="">World Map</h3>
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

export default function AirbnbPriceMap (props){
  const data = []
  Object.keys(city_locations).forEach(async (city_name) => {
    const req = await fetch(`https://shelbernstein.ngrok.io/data/airbnb/apt/${city_name}`);
    const json = await req.json()
    for (let key of json) {
      data.push(
        json[key]
      )
    }
  })


 const otherMaps = Object.entries(props.data).map((pair) => {
    return <JustMap title={pair[0]} data={pair[1]} left={0} />
  })

  return ( <>
    <AirbnbWorldMap></AirbnbWorldMap>
  
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