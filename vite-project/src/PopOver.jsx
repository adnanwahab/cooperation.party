import { Fragment, useState, useEffect} from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
function ListingDetails(props) {
    const listing = props.listing;
    return (
      <div className="listing-details text-black">
        <img src="airbnb.webp"/>
        <div>
            is this a good deal?
            yes it is <span className="text-green-500">20% below similar appartments in the area</span>
        </div>
        <div>Name: {listing.name}</div>
        <div>Host ID: {listing.host_id}</div>
        <div>Host Name: {listing.host_name}</div>
        <div>Neighbourhood Group: {listing.neighbourhood_group}</div>
        <div>Neighbourhood: {listing.neighbourhood}</div>
        <div>Latitude: {listing.latitude}</div>
        <div>Longitude: {listing.longitude}</div>
        <div>Room Type: {listing.room_type}</div>
        <div>Price: ${listing.price}</div>
        <div>Minimum Nights: {listing.minimum_nights}</div>
        <div>Number of Reviews: {listing.number_of_reviews}</div>
        <div>Last Review Date: {listing.last_review}</div>
        <div>Reviews Per Month: {listing.reviews_per_month}</div>
        <div>Host Listings Count: {listing.calculated_host_listings_count}</div>
        <div>Availability (365 Days): {listing.availability_365}</div>
        <div>Number of Reviews (Last 12 Months): {listing.number_of_reviews_ltm}</div>
        <div>License: {listing.license}</div>
      </div>
    );
  }
  
  // Usage
  const listingData = {
      "name": "Home in Columbus · ★4.81 · 3 bedrooms · 3 beds · 2 baths",
      // ... other data ...
      "license": "2022-2475"
  };
  
  



export default function Example(props) {
    const {open, setOpen} = props;
    const [listingDetails, setListingDetails] = useState({})

    useEffect(() => {
        async function fetchData() {
            if (! open) return
            const req = await fetch(`https://shelbernstein.ngrok.io/get_apt_details?apt_id=${open}`);
            const json = await req.json();
            setListingDetails(json)
        }
        fetchData()
    }, [open])


  return (
    <Transition.Root show={!! open} as={Fragment}>
      <Dialog as="div" className="relative z-10" onClose={setOpen}>
        <div className="fixed inset-0" />

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500 sm:duration-700"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500 sm:duration-700"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                  <div className="flex h-full flex-col overflow-y-scroll bg-white py-6 shadow-xl">
                    <div className="px-4 sm:px-6">
                      <div className="flex items-start justify-between">
                        <Dialog.Title className="text-base font-semibold leading-6 text-gray-900">
                          Listing Detail
                        </Dialog.Title>
                        <div className="ml-3 flex h-7 items-center">
                          <button
                            type="button"
                            className="relative rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                            onClick={() => setOpen(false)}
                          >
                            <span className="absolute -inset-2.5" />
                            <span className="sr-only">Close panel</span>
                            <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                          </button>

                        </div>
                      </div>
                    </div>
                    <div className="relative mt-6 flex-1 px-4 sm:px-6">
                    <ListingDetails listing={listingDetails} />
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  )
}