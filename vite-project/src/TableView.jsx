import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/20/solid'
import {useEffect, useState} from 'react'
import MapView from './MapView'
import BarChart from './BarChart';
import {cellToLatLng} from 'h3-js'

let cool_beans = [
  [
      "Mobile World Conference",
      "Spain",
      "Barcelona",
      "January 24-27"
  ],
  [
      "IBM Think 2020",
      "US",
      "San Francisco",
      "February 5-7"
  ],
  [
      "Cisco Live Melbourne",
      "Australia",
      "Melbourne",
      "March 3-6 "
  ],
  [
      "Salesloft’s Rev2020",
      "US",
      "San Francisco",
      "April 9-11"
  ],
  [
      "SaaStr",
      "US",
      "San Jose",
      "June 10-12"
  ],
  [
      "Microsoft WSLConf",
      "US",
      "Shanghai",
      "July 10-11 "
  ],
  [
      "Facebook Global Marketing Summit",
      "US",
      "Singapore",
      "August 9-12"
  ],
  [
      "Concur Fusion",
      "US",
      "Taiwan",
      "September 9-12"
  ],
  [
      "Qualtrics X4",
      "Japan",
      "Tokyo",
      "October 10-12"
  ],
  [
      "Intel Labs Day",
      "US",
      "San Francisco",
      "November-12"
  ],
  [
      "SXSW",
      "US",
      "Austin",
      "December 12-22"
  ],
  [
      "Microsoft MVP Global Summit",
      "US",
      "Bellevue",
      "January 15-20"
  ],
  [
      "F5 Agility 2020",
      "US",
      "Orlando",
      "February 16-19"
  ],
  [
      "SAP Ariba Live",
      "US",
      "Las Vegas",
      "March 16-18"
  ],
  [
      "Game Developers Conference",
      "US",
      "San Francisco",
      "April 16-20"
  ],
  [
      "Domopalooza",
      "US",
      "Salt Lake City",
      "May 12-20 "
  ],
  [
      "Nvidia’s GTC",
      "US",
      "San Jose",
      "June 22-26"
  ],
  [
      "EmTech Asia",
      "Singapore",
      "Singapore ",
      "July 24-26"
  ],
  [
      "Oracle Modern Business Experience 2020",
      "US",
      "Chicago",
      "August 23-26 "
  ],
  [
      "Adobe Summit",
      "US",
      "Las Vegas",
      "September-29"
  ],
  [
      "Okta’s Oktane 2020",
      "US",
      "San Francisco",
      "October 30-April 2"
  ],
  [
      "Atlassian Summit 2020",
      "US",
      "Las Vegas",
      "November 31-April 2"
  ],
  [
      "Google Cloud Next ‘20",
      "US",
      "San Francisco",
      "December 6-8 "
  ],
  [
      "Outreach Unleash",
      "US",
      "San Diego",
      "January 7-9"
  ],
  [
      "Facebook F8 Conference",
      "US",
      "San Jose",
      "February 5-6"
  ],
  [
      "Shopify Unite",
      "Canada",
      "Toronto",
      "March 6-8 "
  ],
  [
      "Google I/O",
      "US",
      "Mountain View",
      "April 12-14"
  ],
  [
      "TNW 2020",
      "Netherlands",
      "Amsterdam",
      "May 18-19"
  ],
  [
      "CERAWeek 2020",
      "US",
      "Houston",
      "June 9-13"
  ],
  [
      "ICLR 2020",
      " Ethiopia.",
      "Addis Ababa",
      "July 26-30"
  ],
  [
      "Collision",
      "Canada",
      "Toronto",
      "August 22-25"
  ]
]


const tabs = [
    { name: 'My Account', href: '#', current: false },
    { name: 'Company', href: '#', current: false },
    { name: 'Team Members', href: '#', current: true },
    { name: 'Billing', href: '#', current: false },
  ]
  
  function classNames(...classes) {
    return classes.filter(Boolean).join(' ')
  }




function Steps({options, setSelectedValue, selectedValue}) {

  const steps = options.map ((_, index) => 
    {
      return     { id: _, 
        
        
        name: cool_beans[index].join('\n'), 
        
        
        href: '#', status: 
      
      
    selectedValue === _ ? 'current'  : 'upcoming'
    
    }

    }
    //{ id: 'Step 2', name: 'Application form', href: '#', status: 'upcoming' },
    //{ id: 'Step 3', name: 'Preview', href: '#', status: 'upcoming' },
  )
  
  const onClick = (index) => {

    steps[index].upcoming = 'current'

    steps[index].status = 'current'
    setSelectedValue(options[index])
  }
    return (
      <nav aria-label="Progress">
        <ol role="list" className="space-y-4 md:flex md:space-x-8 md:space-y-0">
          {steps.map((step, index) => (
            <li key={step.name} className="md:flex-1" onClick={() => onClick(index)}>
              {step.status === 'complete' ? (
                <a
                  href={step.href}
                  className="group flex flex-col border-l-4 border-indigo-600 py-2 pl-4 hover:border-indigo-800 md:border-l-0 md:border-t-4 md:pb-0 md:pl-0 md:pt-4"
                >
                  <span className="text-sm font-medium text-indigo-600 group-hover:text-indigo-800">{step.id}</span>
                  <span className="text-sm font-medium">{step.name}</span>
                </a>
              ) : step.status === 'current' ? (
                <a
                  href={step.href}
                  className="flex flex-col border-l-4 border-indigo-600 py-2 pl-4 md:border-l-0 md:border-t-4 md:pb-0 md:pl-0 md:pt-4"
                  aria-current="step"
                >
                  <span className="text-sm font-medium text-indigo-600">{step.id}</span>
                  <span className="text-sm font-medium">{step.name}</span>
                </a>
              ) : (
                <a
                  href={step.href}
                  className="group flex flex-col border-l-4 border-gray-200 py-2 pl-4 hover:border-gray-300 md:border-l-0 md:border-t-4 md:pb-0 md:pl-0 md:pt-4"
                >
                  <span className="text-sm font-medium text-gray-500 group-hover:text-gray-700">{step.id}</span>
                  <span className="text-sm font-medium">{step.name}</span>
                </a>
              )}
            </li>
          ))}
        </ol>
      </nav>
    )
  }

  const SelectComponent = (props) => {
    return <Steps
    {...props}
    ></Steps>
    // return 
    //   <div>
    //     <select
    //     className='city-selector'
    //       value={selectedValue}
    //       onChange={e => setSelectedValue(e.target.value)}
    //     >
    //       <option value="" disabled>Select an option</option>
    //       {options.map((option, index) => (
    //         <option key={index} value={option}>
    //           {option}
    //         </option>
    //       ))}
    //     </select>
    //   </div>
    // );
  };
  
 function Tabs({tabs}) {
    return (
      <div>
        <div className="sm:hidden">
          <label htmlFor="tabs" className="sr-only">
            Select a tab
          </label>
          {/* Use an "onChange" listener to redirect the user to the selected tab URL. */}
          <select
            id="tabs"
            name="tabs"
            className="block w-full rounded-md border-gray-300 focus:border-indigo-500 focus:ring-indigo-500"
            defaultValue={tabs.find((tab) => tab.current).name}
          >
            {tabs.map((tab) => (
              <option key={tab.name}>{tab.name}</option>
            ))}
          </select>
        </div>
        <div className="hidden sm:block">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex" aria-label="Tabs">
              {tabs.map((tab) => (
                <a
                  key={tab.name}
                  href={tab.href}
                  className={classNames(
                    tab.current
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
                    'w-1/4 border-b-2 py-4 px-1 text-center text-sm font-medium'
                  )}
                  aria-current={tab.current ? 'page' : undefined}
                >
                  {tab.name}
                </a>
              ))}
            </nav>
          </div>
        </div>
      </div>
    )
  }

const people = [
  { name: 'Lindsay Walton', title: 'Front-end Developer', email: 'lindsay.walton@example.com', role: 'Member' },
]

const isAscending = [false, true, false]

export default function Example(props) {
  const [getNeighborhoodDetails, setNeighborhoodDetails] = useState([])
  const [schedulePlaces, setSchedulePlaces] = useState([])
  const tabs = Object.keys(props.data).map((_, i) => {
    return _
})
  const [selectedCity, setSelectedCity] = useState(tabs[0]);
  const [selectedApt, setSelected] = useState('')

  console.log('tableview props', props)
  //routes -> render on map
  //land usage

  useEffect(function () {
    //console.log(selectedApt)
      const getNeighborhoodDetails = async function () {
        let url  = 'https://pypypy.ngrok.io/neighborhoodDetails'
          let res = await fetch(url, {
          method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              apt_url: selectedApt[0],
              schedule_text: document.querySelector('textarea').value.split('\n')[1],
              city_name: selectedCity
            })
          })
          let json = await res.json()
          setNeighborhoodDetails(json.key)
          setSchedulePlaces(Object.keys(json.schedule_places))
          console.log('json.key', json.schedule_places)
      }
      if (selectedApt && selectedApt[0])
        getNeighborhoodDetails()
  }, [selectedApt]);



    const keys = ['id', 'price',  'good_deal', 'commute_distance']
    let [sortBy, setSortBy] = useState(keys[0])
    let data = 
    Object.entries(props.data[selectedCity])
    .sort(function (one, two) {
        let dir = keys[keys.indexOf(sortBy)] ? -1 : 1
        return (one[1][sortBy] - two[1][sortBy]) //* (dir)
    }).slice(0, 10)

    window.addToPdf = function (city) {
      return props.data[city]
    }

    function toggleSortBy(_) {
        if (_) return keys[keys.indexOf(_)] = ! keys[keys.indexOf(_)];
        setSortBy(_)
    }
   
    //complaint_num - h3
    //commute-to-PoI //library + rock climbing + wind surfing
    const places = schedulePlaces || ['yoga', 'rock_climbing', 'coworking', 'kite']
    const selectedIndex = tabs.indexOf(selectedCity);
    const routes = props.routes[selectedIndex]
    const barChartData = routes?.map((_, i) => {
      return {letter: Object.keys(props.schedule)[i], frequency:_.routes[0].duration }
  });

    let centroid = [0,0]
    Object.keys(props.h3_hexes[selectedCity]).map(_ => {
      return cellToLatLng(_)
    }).forEach(_ => { 
      centroid[0] += _[0]
      centroid[1] += _[1]
    })

    let len = Object.keys(props.h3_hexes[selectedCity]).length

    centroid[0] /= len
    centroid[1] /= len

  return (
    <>
    <SelectComponent 
    selectedValue={selectedCity} 
    setSelectedValue={setSelectedCity}
    options={tabs}></SelectComponent>
    <BarChart data={barChartData} city={selectedCity} schedule={Object.keys(props.schedule).join('')}></BarChart>
    <div class="relative h-96 w-96">
      <MapView centroid={centroid} left={0} routes={props.routes[selectedIndex]}  data={Object.values(props.data[selectedCity])}/>
      <MapView centroid={centroid} left={600} h3_hexes={props.h3_hexes[selectedCity]} routes={getNeighborhoodDetails}  data={Object.values(props.data[selectedCity])}/>
    </div>
    <div className="mt-20 px-4 sm:px-6 lg:px-8">
      <div className="sm:flex sm:items-center">
        <div className="mt-4 sm:ml-16 sm:mt-0 sm:flex-none">
        </div>
      </div>
      <div className="mt-8 flow-root">
        <div className="-mx-4 -my-2 overflow-x-auto sm:-mx-6 lg:-mx-8">
          <div className="inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
            <table className="min-w-full divide-y divide-gray-300">
              <thead>
                <tr>
                  {keys.map((_, index) => {
                    return (<th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    <a href="#" className="group inline-flex">
                      {_}
                      <span className="ml-2 flex-none rounded bg-gray-100 text-gray-900 group-hover:bg-gray-200">
{isAscending[index] ? <ChevronDownIcon className="h-5 w-5" aria-hidden="true" onClick={() => setSortBy(_)}/> : 

<ChevronUpIcon className="h-5 w-5" aria-hidden="true" onClick={() => setSortBy(_) }/>

}
                      </span>
                    </a>
                  </th>)
                  })}
                  {/* <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    <a href="#" className="group inline-flex">
                     {keys[1]}
                      <span className=" ml-2 flex-none rounded text-gray-400 group-hover:visible group-focus:visible">
                        <ChevronDownIcon
                          className=" ml-2 h-5 w-5 flex-none rounded text-gray-400 group-hover:visible group-focus:visible"
                          aria-hidden="true"
                        />
                      </span>
                    </a>
                  </th>
                  <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                    <a href="#" className="group inline-flex">
                      {keys[2]}
                      <span className=" ml-2 flex-none rounded text-gray-400 group-hover:visible group-focus:visible">
                        <ChevronDownIcon
                          className=" ml-2 h-5 w-5 flex-none rounded text-gray-400 group-hover:visible group-focus:visible"
                          aria-hidden="true"
                        />
                      </span>
                    </a>
                  </th> */}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {data.map((pair) => {
                    let person = pair[1]
                    
                    return (
                  <tr key={person.email} class="hover:bg-green-500" onMouseEnter={_ => setSelected(pair)}>
                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-0">
                    <a target="_blank" href={`https://airbnb.com/rooms/${person[keys[0]]}`}>https://airbnb.com/rooms/{person[keys[0]]}</a>
                    </td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[1]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[2]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[3]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[4]]}</td>

                  </tr>
                ) } )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
    </>
  )
}