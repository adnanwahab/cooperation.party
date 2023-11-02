//IsochroneMap
const isGeoCoordinate = (pair) => {
    return Array.isArray(pair) && parseFloat(pair[0][0]) && parseFloat(pair[0][1])
  }
function Isochronemap(props) {
  if (! props.reports) return <>not found sorry 🐻</>
  console.log(props)
  let _ = props.reports.map((_, idx) => {
    const reasons = _['reasoning_explanation']
    return <div key={idx}>
            <div>{_['name'] + '  ' + _.house.url}</div>
            <>Other Houses Within Neighborhood{List(_.houses_within_suggested_neighborhood.map(_ => _.url))}</>
            <div>{reasons.split('\n').map(_ => <div>{_}</div>)}</div> 
            <div>
               <BarChart data={Object.entries(_['commutes']).map(_ => {
                  return { letter: _[0], frequency: parseFloat(_[1].replace('mi', '')) }
                })}></BarChart>
            </div>
          </div>
  })
  return <>
  <div className="text-xl">{props.city}</div>
    <HexagonMap {...props}></HexagonMap>
    {[...Array(8).keys()].map(_ => <br key={_}/>)}
    {_}
  </>
}