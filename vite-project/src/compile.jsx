import TableView from './TableView'
import TrafficMap from './TrafficMap'
import HexagonWorld from './hexagon_world';
import AirbnbPriceMap from './AirbnbPriceMap'
import AnabelleMap from './AnabelleMap'


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
      <label className="text-base font-semibold text-gray-300">Clarifications</label>
      <p className="text-sm text-gray-500">What type of event do you prefer?</p>
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


const List = (list) => 
  (<ul className="overflow-scroll h-64">
    <li key="item font-white">{list.length}</li>
    {list.map((item, idx) => <li key={idx}>{item}</li>)}
  </ul>)

function GeoCoder ({onChange}) {

    return <>
           <label>How much money will my house make on airbnb?</label>
           <input onChange={onChange} type="text" class="text-black"></input>
           </>
  }
  
  function EarningsCalculator({address}) {
    //goal make 1,000 millionaires in 30-90 days
    //return <>{address ? '1 million' : 'idk depends on where you live'}</>
  }
  
  function TextPresenter(props) {
    return <div className=" border border-purple-500">
      {props.text}
    </div>
  }
  
  function isIsochroney(datum) {
    return datum[0] && datum[0][0] && datum[0][0][1] && datum[0][0][1].type === 'node'
  }
  



export default function compile (dataList, apply_) {
//  let [getSelected, setSelected] = useState('')
  //console.log('dataList', dataList)
  //const [address, setAddress] = useState('')

  const result = dataList.map(function (datum, index) {
    if ('<AnabelleMap>' === datum.component) {
      return <AnabelleMap data={datum.data} />
    }

    if ('<airbnb_price_map>' === datum.component) {
      return <AirbnbPriceMap data={datum.data} />
    }

    if (datum.component === '<trafficMap>') {
      return <TrafficMap></TrafficMap>
    }
    if (datum.component === '<GeoCoder>') {
      return <GeoCoder onChange={(e) => setAddress(e.target.value)}/>
    }

    if (datum.component === '<EarningsCalculator>') {
      return <EarningsCalculator setAddress="onChange"/>

    }

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