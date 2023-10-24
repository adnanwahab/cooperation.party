const List = (list) => 
  (<ul className="overflow-scroll h-64">
    <li key="item font-white">{list.length}</li>
    {list.map((item, idx) => <li key={idx}>{item}</li>)}
  </ul>)

function geoCoder ({onChange}) {
    return <input onChange={onChange} type="text"></input>
  }
  
  function earningsCalculator({address}) {
    //goal make 1,000 millionaires in 30-90 days
    return <>{address ? '1 million' : 'idk depends on where you live'}</>
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
  //console.log(dataList)
  //const [address, setAddress] = useState('')

  const result = dataList.map(function (datum, index) {
    if (datum.component === '<geocoder>') {
      return <geoCoder onChange={(e) => setAddress(e.target.value)}/>
    }

    if (datum.component === '<earningsCalculator>') {
      return <earningsCalculator setAddress="onChange"/>

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