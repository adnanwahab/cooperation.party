import {useRef, useMemo, useState, useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import ReactMap, {Source, Layer, Marker} from 'react-map-gl';
import DeckGL, {GeoJsonLayer, ArcLayer} from 'deck.gl';
import {range} from 'd3-array';
import {scaleQuantile} from 'd3-scale';
import { Listbox } from '@headlessui/react'

import {Runtime, Inspector} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/dist/runtime.js";
import define from "https://api.observablehq.com/@marialuisacp/pie-chart.js?v=3";
import define2 from "https://api.observablehq.com/d/84ce55045edfd14f.js?v=3";

function PlaceHolder() {
return  <>
  {/* <div id="land-use-panel" class="f8 panel sans-serif active" data-nanocomponent="ncid-5327"><div class="pv4 ph5 bb b--black-10"><h3 class="fw6 f4 mt2 mb4">Land Use</h3><p class="lh-copy">Land use data reveals the general distribution of existing functions and the proximity to destinations or community-serving facilities within the area. <span class="ml1"></span><a href="https://morphocode.com/urban-performance-measures/" target="_blank" rel="noopener" class="b--black-20 bb black-60 hover-b--black hover-black link">Learn more</a></p></div><div class="pa4 pb6 pl5 pr4 bb b--black-10 flex flex-wrap"><div class="w-50"><h4 class="fw6 mt0 pv2 mb4">Overview</h4><dl class="ma0"><div class="lh-title mb4"><dd class="f3 fw6 ml0 mb1"><div class="dib mono" id="ncid-7a6e" data-nanocomponent="ncid-7a6e"><span>2,247</span><span class="fw1 pl2"></span></div></dd><dt class="fw4 mono">Lots</dt></div><div class="lh-title mb4"><dd class="f3 fw6 ml0 mb1"><div class="dib mono" id="ncid-833b" data-nanocomponent="ncid-833b"><span>335.04</span><span class="fw1 pl2">ac</span></div></dd><dt class="fw4 mono">Lot Area</dt></div></dl></div><div class="w-50"><div class="flex items-center justify-between lh-copy mb4"><h4 class="fw6 lh-copy ma0 pv2">Entropy Score</h4><a href="#" target="_blank" rel="noopener" aria-label="Learn more" class="br-100 dib glow h1-extra hide-child icon-info link o-70 relative w1-extra" id="ncid-ec05" data-nanocomponent="ncid-ec05"> <div class="absolute child right-0 top-50 bg-dark-gray br2 shadow-2 pe-none white pa4 lh-copy info-tooltip">
              Quantifies the diversity of land uses. The index varies between 0 and 1, where 0 represents a single land use, and 1 represents a maximum land-use mix</div></a></div><div class="mono" id="ncid-7e2a" data-nanocomponent="ncid-7e2a" data-onloadid1h7tf="o10"><svg width="150" height="150"><g transform="translate(75,75)"><text class="f3 fw6" text-anchor="middle" dy="0.35em" fill="currentColor">0.76</text><path fill="#f9eddb" d="M-4.095285467891279,-74.88810744661984A75,75,0,0,1,-1.3777276490407723e-14,-75L-9.184850993605149e-15,-50A50,50,0,0,0,-2.7301903119275197,-49.925404964413225Z"></path><path fill="#f6d9cb" d="M-24.726930934510147,-70.80663024434905A75,75,0,0,1,-4.095285467891279,-74.88810744661984L-2.7301903119275197,-49.925404964413225A50,50,0,0,0,-16.484620623006766,-47.20442016289936Z"></path><path fill="#f1b89c" d="M-73.87532297467851,-12.939731658227549A75,75,0,0,1,-24.726930934510147,-70.80663024434905L-16.484620623006766,-47.20442016289936A50,50,0,0,0,-49.25021531645235,-8.626487772151698Z"></path><path fill="#df7649" d="M-17.869600520688618,72.84008084311141A75,75,0,0,1,-73.87532297467851,-12.939731658227549L-49.25021531645235,-8.626487772151698A50,50,0,0,0,-11.91306701379241,48.5600538954076Z"></path><path fill="#cf4f4f" d="M63.55679998877113,-39.81875406372408A75,75,0,0,1,-17.869600520688618,72.84008084311141L-11.91306701379241,48.5600538954076A50,50,0,0,0,42.371199992514086,-26.54583604248272Z"></path><path fill="#A9AECA" d="M62.95300133821397,-40.76664840909575A75,75,0,0,1,63.55679998877113,-39.81875406372408L42.371199992514086,-26.54583604248272A50,50,0,0,0,41.968667558809315,-27.177765606063836Z"></path><path fill="#bec6cc" d="M60.93923700371658,-43.71966826732404A75,75,0,0,1,62.95300133821397,-40.76664840909575L41.968667558809315,-27.177765606063836A50,50,0,0,0,40.626158002477716,-29.146445511549356Z"></path><path fill="#bde7f4" d="M28.114434535616876,-69.53113382321989A75,75,0,0,1,60.93923700371658,-43.71966826732404L40.626158002477716,-29.146445511549356A50,50,0,0,0,18.742956357077915,-46.354089215479924Z"></path><path fill="#A3D393" d="M11.232164405987957,-74.15415350981263A75,75,0,0,1,28.114434535616876,-69.53113382321989L18.742956357077915,-46.354089215479924A50,50,0,0,0,7.488109603991972,-49.436102339875085Z"></path><path fill="#8DA2B4" d="M5.9694374655553775,-74.76206134360412A75,75,0,0,1,11.232164405987957,-74.15415350981263L7.488109603991972,-49.436102339875085A50,50,0,0,0,3.9796249770369183,-49.84137422906941Z"></path><path fill="#E4E4E4" d="M2.105148875607522,-74.97044983332786A75,75,0,0,1,5.9694374655553775,-74.76206134360412L3.9796249770369183,-49.84137422906941A50,50,0,0,0,1.403432583738348,-49.98029988888524Z"></path><path fill="#F9F9F9" d="M4.592425496802574e-15,-75A75,75,0,0,1,2.105148875607522,-74.97044983332786L1.403432583738348,-49.98029988888524A50,50,0,0,0,3.061616997868383e-15,-50Z"></path></g></svg></div></div></div><div class="pa4 pb6 ph5 bb b--black-10"><h4 class="fw6 mt0 pv2 mb4">Land Use Categories</h4><ul class="list ma0 mono pa0" id="ncid-e841" data-nanocomponent="ncid-e841" data-onloadid1h7tf="o11"><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(249, 237, 219); width: 12px; height: 12px;"></span><span class="_label flex-auto">One &amp; Two Family Buildings</span><span class="_value">2.9</span><span class="_unit fw1 nr1 o-70" style="display: inline-block;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 2%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(246, 217, 203); width: 12px; height: 12px;"></span><span class="_label flex-auto">Multi-Family Walk-Up Buildings</span><span class="_value">15.0</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 12%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(241, 184, 156); width: 12px; height: 12px;"></span><span class="_label flex-auto">Multi-Family Elevator Buildings</span><span class="_value">56.6</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 45%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(223, 118, 73); width: 12px; height: 12px;"></span><span class="_label flex-auto">Residential &amp; Commercial Mix</span><span class="_value">80.2</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 63%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(207, 79, 79); width: 12px; height: 12px;"></span><span class="_label flex-auto">Commercial &amp; Office</span><span class="_value">126.4</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 100%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(169, 174, 202); width: 12px; height: 12px;"></span><span class="_label flex-auto">Industrial &amp; Manufacturing</span><span class="_value">0.8</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 1%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(190, 198, 204); width: 12px; height: 12px;"></span><span class="_label flex-auto">Transportation &amp; Utility</span><span class="_value">2.5</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 2%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(189, 231, 244); width: 12px; height: 12px;"></span><span class="_label flex-auto">Public Facilities &amp; Institutions</span><span class="_value">30.1</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 24%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(163, 211, 147); width: 12px; height: 12px;"></span><span class="_label flex-auto">Open Space &amp; Outdoor Recreation</span><span class="_value">12.5</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 10%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(141, 162, 180); width: 12px; height: 12px;"></span><span class="_label flex-auto">Parking Facilities</span><span class="_value">3.8</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 3%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(228, 228, 228); width: 12px; height: 12px;"></span><span class="_label flex-auto">Vacant Land</span><span class="_value">2.8</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 2%;"></span></div></li><li class="mb2 pb1"><div class="f8 flex items-center pv2"><span class="br-100 mr2 dib" style="display: inherit; background-color: rgb(249, 249, 249); width: 12px; height: 12px;"></span><span class="_label flex-auto">No Data</span><span class="_value">1.5</span><span class="_unit fw1 nr1 o-70" style="display: none;">ac</span></div><div class="bg-light-gray overflow-hidden h-tiny"><span class="_bar bg-near-black h-100 db" style="width: 1%;"></span></div></li></ul></div><div class="b--black-10 bb flex flex-wrap ph4 pv4"><h4 class="fw6 lh-copy mb4 mt0 pa2 w-50">Parks &amp; Open space</h4><h4 class="fw6 lh-copy mb4 mt0 pa2 w-50">Vacant Land</h4><div class="mb4 pa2 w-50"><div class="f3 fw6 mono" id="ncid-dbaf" data-nanocomponent="ncid-dbaf" data-onloadid1h7tf="o12"><svg width="120" height="120"><g transform="translate(60,60)"><text text-anchor="middle" dy="0.35em" fill="currentColor">3.7%</text><path fill="#4A4A4A" d="M3.67394039744206e-15,-60A60,60,0,0,1,13.907380991557929,-58.36595543598728L10.430535743668447,-43.77446657699046A45,45,0,0,0,2.7554552980815448e-15,-45Z"></path><path fill="#E5E5E5" d="M13.907380991557929,-58.36595543598728A60,60,0,1,1,-1.1021821192326178e-14,-60L-8.266365894244634e-15,-45A45,45,0,1,0,10.430535743668447,-43.77446657699046Z"></path></g></svg></div></div><div class="mb4 pa2 w-50"><div class="f3 fw6 mono" id="ncid-adf7" data-nanocomponent="ncid-adf7" data-onloadid1h7tf="o13"><svg width="120" height="120"><g transform="translate(60,60)"><text text-anchor="middle" dy="0.35em" fill="currentColor">0.8%</text><path fill="#4A4A4A" d="M3.67394039744206e-15,-60A60,60,0,0,1,3.0948921884695335,-59.92012718896506L2.3211691413521502,-44.940095391723794A45,45,0,0,0,2.7554552980815448e-15,-45Z"></path><path fill="#E5E5E5" d="M3.0948921884695335,-59.92012718896506A60,60,0,1,1,-1.1021821192326178e-14,-60L-8.266365894244634e-15,-45A45,45,0,1,0,2.3211691413521502,-44.940095391723794Z"></path></g></svg></div></div></div></div>

 */}
 return (
        <div id="land-use-panel" className="f8 panel sans-serif active">
            <div className="pv4 ph5 bb b--black-10">
                <h3 className="fw6 f4 mt2 mb4">Land Use</h3>
                <p className="lh-copy">
                    Land use data reveals the general distribution of existing functions and the proximity to destinations or community-serving facilities within the area.
                    <span className="ml1"></span>
                    <a href="https://morphocode.com/urban-performance-measures/" target="_blank" rel="noopener" className="b--black-20 bb black-60 hover-b--black hover-black link">Learn more</a>
                </p>
            </div>

            <div className="pa4 pb6 pl5 pr4 bb b--black-10 flex flex-wrap">
                <div className="w-50">
                    <h4 className="fw6 mt0 pv2 mb4">Overview</h4>
                    <dl className="ma0">
                        <div className="lh-title mb4">
                            <dd className="f3 fw6 ml0 mb1"><span>2,247</span><span className="fw1 pl2"></span></dd>
                            <dt className="fw4 mono">Lots</dt>
                        </div>
                        <div className="lh-title mb4">
                            <dd className="f3 fw6 ml0 mb1"><span>335.04</span><span className="fw1 pl2">ac</span></dd>
                            <dt className="fw4 mono">Lot Area</dt>
                        </div>
                    </dl>
                </div>
                <div className="w-50">
                    <div className="flex items-center justify-between lh-copy mb4">
                        <h4 className="fw6 lh-copy ma0 pv2">Entropy Score</h4>
                        <a href="#" target="_blank" rel="noopener" aria-label="Learn more" className="br-100 dib glow h1-extra hide-child icon-info link o-70 relative w1-extra">
                            <div className="absolute child right-0 top-50 bg-dark-gray br2 shadow-2 pe-none white pa4 lh-copy info-tooltip">
                                Quantifies the diversity of land uses. The index varies between 0 and 1, where 0 represents a single land use, and 1 represents a maximum land-use mix
                            </div>
                        </a>
                    </div>
                    {/* The SVG component and its content go here */}
                </div>
            </div>

            <div className="pa4 pb6 ph5 bb b--black-10">
                <h4 className="fw6 mt0 pv2 mb4">Land Use Categories</h4>
                {/* The list of land use categories goes here */}
            </div>

            <div className="b--black-10 bb flex flex-wrap ph4 pv4">
                <h4 className="fw6 lh-copy mb4 mt0 pa2 w-50">Parks &amp; Open space</h4>
                <h4 className="fw6 lh-copy mb4 mt0 pa2 w-50">Vacant Land</h4>
                {/* The rest of the content for Parks & Open space and Vacant Land goes here */}
            </div>
        </div>

        {[...Array(6).keys()].map(_ => <img src={`/${_}.png`} /> )}
        
    );

  </>
}


function BarChart(props) {
  console.log(props.data)
  const chartRef = useRef();
  console.log('BAR CHART')
  useEffect(() => {
    const runtime = new Runtime();
    runtime.module(define2, name => {
      if (name === "chart") return new Inspector(chartRef.current);
    }).redefine('alphabet', props.data)
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={chartRef} />
    </>
  );
}


function Chart (props) { 
  console.log(props.getCoefficents)
  const chartRef = useRef();
  useEffect(() => {
    const runtime = new Runtime();
    new Runtime().module(define, name => {
      if (name === "chart") return new Inspector(chartRef.current);
    }).redefine('dataset', 
    props.getCoefficents
    //({ 'A': 35 * Math.random(), 'B': 30, 'C':25, 'D': 20, 'E': 15, 'F': 10})
    )
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={chartRef} />
    </>
  );
}



const apt_choice = [35, 139]

const typesOfPlaces = [
  { id: 1, name: 'coffee', unavailable: false },
  { id: 2, name: 'Bar', unavailable: false },
  { id: 3, name: 'Library', unavailable: false },
  { id: 4, name: 'Grocery', unavailable: true },
  { id: 5, name: 'yoga', unavailable: false },
]

function MyListbox() {
  const [selectedPerson, setSelectedPerson] = useState(typesOfPlaces[0])

  return (
    <Listbox value={selectedPerson} onChange={setSelectedPerson}>
      <Listbox.Button>{selectedPerson.name}</Listbox.Button>
      <Listbox.Options>
        {typesOfPlaces.map((person) => (
          <Listbox.Option
            key={person.id}
            value={person}
            disabled={person.unavailable}
          >
            {person.name}
          </Listbox.Option>
        ))}
      </Listbox.Options>
    </Listbox>
  )
}

async function getIsochrone(longitude, latitude, contours_minutes) {
  const accessToken = 'your-access-token-here';
  const isochroneUrl = `https://api.mapbox.com/isochrone/v1/mapbox/driving-traffic/${longitude}%2C${latitude}?contours_minutes=${contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=${accessToken}`;

  try {
      const response = await fetch(isochroneUrl);
      
      if (response.ok) {
        const geojsonData = await response.json();
          console.log(geojsonData);
          // Do something with the geojson data
      } else {
          console.log('Failed to fetch isochrone data:', response.status, response.statusText);
      }
  } catch (error) {
      console.error('Error fetching isochrone data:', error);
  }
}
// Call the function
const colors = [
  [0, '#3288bd'],
  [1, '#66c2a5'],
  [2, '#abdda4'],
  [3, '#e6f598'],
  [4, '#ffffbf'],
  [5, '#fee08b'],
  [6, '#fdae61'],
  [7, '#f46d43'],
  [8, '#d53e4f']
]

const dataLayer = () => {
  let color = colors[Math.floor(Math.random() * 8)][1]
  console.log(color)
  return {
  id: 'data',
  type: 'fill',
  paint: {
    'fill-color': color,
    'fill-opacity': 0.8
  }
}}

async function fetchCoffeeShops() {
  const latitude = 37.8;
  const longitude = -122.4;
  const radius = 1000; // radius in meters

  // Build Overpass API URL
  const query = `[out:json][timeout:25];` +
                `(node["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01});` +
                `way["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01});` +
                `relation["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01}););` +
                `out body;`;
  const overpassUrl = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;
  
  try {
    const response = await fetch(overpassUrl);
    
    if (response.ok) {
      const data = await response.json();
      const coffeeShops = data.elements;
      console.log("Coffee Shops:", coffeeShops);
      // Do something with the fetched coffee shops
    } else {
      console.log('Failed to fetch data:', response.status, response.statusText);
    }
  } catch (error) {
    console.error('Error fetching coffee shops:', error);
  }
}

function Map(props) {
  console.log(props)
  const mapRef = useRef();
  const [optimalHouse, setOptimalHouse] = useState('')

  if (props.data === 'hello world' ||
  props.data === 'hello-world'
  ) return <></>

  // const geoJson = props.data[0]
  // const coffeeShops = props.data[0][0]
  const latitude = props.data[1][2]
  const longitude = props.data[1][3]

  const onClick = () => {
    console.log('hello')
  }

  const renderShop = (shop, index)=> {
    return <Marker
    key={`marker-${index}`}
    longitude={shop.lon}
    latitude={shop.lat}
    anchor="bottom"
    onClick={e => {
      // If we let the click event propagates to the map, it will immediately close the popup
      // with `closeOnClick: true`
      e.originalEvent.stopPropagation();
      // setPopupInfo(city);
    }}
  >
    {/* <Pin /> */}
  </Marker>
  }

  function merge (json) {
    return json.reduce((acc, item) => {
      acc.features.push(item.features[0])
      return acc
    })
  }

  let geoJson = props.data.map((listing, idx) => {
    return listing[1]
  })

  geoJson = merge(geoJson)

  let renderGeoJson = (<Source type="geojson" data={geoJson}>
      <Layer {...dataLayer()} />
    </Source>)

  let shopMarkers = props.data.map(listing => {
    return listing[0].map(renderShop)
  })

let places = ['commuteDistance', 'library', 'bar', 'coffee']
let placeCoefficents = {} 
places.forEach(place => {
  placeCoefficents[place] = .5
})
//sentences + componentData = list of component + states from props
//makeFn - for now just re-render and redo whole document - con might be slow w/o caching??
//CallFn - change a UI component 
const makeOnCoefficentChange = (name) => {
  return (e) => {
    placeCoefficents[name] = parseFloat(e.target.value) / 100
    setCoefficents(Object.assign({}, placeCoefficents))
  }
}
let coefficentsSliders = places.map((pair) => {
  return <><label>{pair}</label><input type='range' onChange={ makeOnCoefficentChange(pair)}/></>
})
let [getCoefficents, setCoefficents] = useState(placeCoefficents)

useEffect(() => {
  console.log('fetch to callFN here') //make do makeFN with better caching 
  //sentence defines not an endpoint but a function that can be called from endpoint
  //sentence -> returns a component
  //on interaction -> sends a networkRequest -> /rpc/function_name with json = parameters
  //returns function and then re-renders data
  let fn = async ()=>  {
    let _ = await fetch('https://pypypy.ngrok.io/callFn/', {
      method: 'POST',
      redirect: "follow", // manual, *follow, error
      referrerPolicy: "no-referrer", 
      mode: "cors", // no-cors, *cors, same-origin
      cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      credentials: 'omit',
      headers: { "Content-Type": "application/json",
      "ngrok-skip-browser-warning": true,
    },
      body: JSON.stringify({ getCoefficents })
    })
    let data = await _.json()

    return data
  }
  fn().then((_) => {
    let evaluation = `
    You selected Coffee Shops as most Important and there are 300 per square mile near this airbnb
    You Selected libraries as not Important and there none near this airbnb
    You selected Bars as somewhat important and there are a handful close to this airbnb by Train
    You selected Bars as important and there is a really good one called Disco Gay Bar thats a 5 min walk
    `



    setOptimalHouse(JSON.stringify(_, null, 2) + evaluation)
  })
}, [getCoefficents])


  return (<>
  {coefficentsSliders}
    <ReactMap
       ref={mapRef}
        mapboxAccessToken="pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"
      initialViewState={{
        longitude,
        latitude,
        zoom: 14
      }}
      style={{width: 1000, height: 500}}
      onClick={onClick}
      // mapStyle="mapbox://styles/mapbox/streets-v3"
      mapStyle="https://api.maptiler.com/maps/streets/style.json?key=D8xiby3LSvsdgkGzkOmN"
    >
    {shopMarkers}
    {renderGeoJson}
    </ReactMap> 
    <>{optimalHouse}</>
    <Chart getCoefficents={getCoefficents}/>
    <BarChart data={
      props.data.map(_ => _[0]).flat().map(_ => {
        function dist(one, two) {
          let dx = one[0] - two[0]
          let dy = one[1] - two[1]
          return dx * dx * dy * dy
        }

        return {
          frequency: dist([_.lat, _.lon], apt_choice), //dist from apt
          letter: _.tags.amenity + _.tags.name
        }
      })
    }></BarChart>

   {/* <PlaceHolder /> */}
    </>);
}

{/* https://morphocode.com/location-time-urban-data-visualization/ */}
export default Map

