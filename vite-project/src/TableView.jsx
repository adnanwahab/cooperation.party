import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/20/solid'
import {useState} from 'react'
import MapView from './MapView'

const tabs = [
    { name: 'My Account', href: '#', current: false },
    { name: 'Company', href: '#', current: false },
    { name: 'Team Members', href: '#', current: true },
    { name: 'Billing', href: '#', current: false },
  ]
  
  function classNames(...classes) {
    return classes.filter(Boolean).join(' ')
  }
  
 function Tabs({tabs}) {
    console.log(tabs)
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
    const keys = ['price',  'good_deal', 'commute_distance', 'num_complaints']
    let [sortBy, setSortBy] = useState(keys[0])
    let data = 
    
    
    Object.entries(props.data['Denver--Colorado--United-States.json'])
    .sort(function (one, two) {
        let dir = keys[keys.indexOf(sortBy)] ? -1 : 1
        //console.log(dir)
return (one[1][sortBy] - two[1][sortBy]) //* (dir)
    }).slice(100)

    function toggleSortBy(_) {
        console.log(_,'_')
        if (_) return keys[keys.indexOf(_)] = ! keys[keys.indexOf(_)];
        setSortBy(_)
    }
    const tabs = Object.keys(props.data).map((_, i) => {
        return {name: _, current: i === 0}
    })
    //complaint_num - h3
    //commute-to-PoI //library + rock climbing + wind surfing
  return (
    <>
    <Tabs tabs={tabs}></Tabs>
    <MapView data={Object.values(props.data['Denver--Colorado--United-States.json'])}/>
  
    <div className="px-4 sm:px-6 lg:px-8">
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
                  <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-0">
                    <a href="#" className="group inline-flex">
                      Name
                    </a>
                  </th>
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
                  <tr key={person.email}>
                    <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-0">
                    <a target="_blank" href={`https://airbnb.com/rooms/${pair[0]}`}>https://airbnb.com/rooms/{pair[0]}</a>
                    </td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[0]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[1]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[2]]}</td>
                    <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">{person[keys[3]]}</td>

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