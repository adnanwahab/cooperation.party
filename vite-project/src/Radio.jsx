function isNestedArray (arr) {  
    return Array.isArray(arr['Asia']) 
  }

function Poll() {
    const [voted, setHasVoted] = useState(false)
    let buttons = <>
        <button onClick={() => vote(0)}>
             {vote_titles[0]}
          </button>
          <button onClick={() => vote(1)}>
          {vote_titles[1]}
          </button>
          <button onClick={() => vote(2)}>
          {vote_titles[2]}
          </button>
          <button onClick={() => vote(3)}>
          {vote_titles[3]}
          </button>
    </>
    let graph = (<iframe width="100%" height="584" frameborder="0"
    src="https://observablehq.com/embed/@d3/bar-chart?cells=chart"></iframe>)
  
    function vote ( ){
      // setHasVoted(true)
    }
    return  <> {voted ? graph : buttons}</>
  }

let hasBeenFlagged = false

function Radio(props) {

  let cities = props.cities

  let default_value = ''

  if (! hasBeenFlagged) {
    if (cities.indexOf('Tokyo, Japan') == -1) return 
    hasBeenFlagged = true
    default_value = 'Tokyo, Japan'
  }

  let [getCity, setCity] = useState(default_value)
  let [_, set_] = useState(false)

  let apply_ = props.apply_

  useEffect(() => {
    setFormData(props.formDataKey, getCity)
    apply_() //dont just send text -> send the data on the client 
    //get all satellite images for every country in america
    //tune contrast -> see if there are any patterns
  }, [getCity])

  return (
    <div>
      <fieldset className="mt-4">
        <div className="space-y-4">
          {cities.map((city, idx) => (
            <div key={idx} className="flex items-center">
              <input
                id={city}
                name="notification-method"
                type="radio"
                checked={getCity === city}
                className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-600"
                onChange={() => {setCity(city)}}
              />
              <label htmlFor={city} className="ml-3 block text-sm font-medium leading-6 text-gray-900">
                {city}
              </label>
            </div>
          ))}
        </div>
      </fieldset>
      -----------------------------------------------------
    </div>
  )
}