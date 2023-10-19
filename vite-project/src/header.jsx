import { useState } from 'react'
import { Dialog } from '@headlessui/react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Product', href: '#' },
  { name: 'Features', href: '#' },
  { name: 'Marketplace', href: '#' },
  { name: 'Company', href: '#' },
]
import { Popover } from '@headlessui/react'
import { jsPDF } from "jspdf";
import { Page, Text, View, Document, StyleSheet } from '@react-pdf/renderer';
import ReactPDF from '@react-pdf/renderer';
import{ Menu, Transition } from '@headlessui/react'
import { Fragment, useEffect, useRef } from 'react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'
const fork = async (e) => {
  let url = `https://pypypy.ngrok.io/share/`
console.log('123')
  let fn = await fetch( url , {
method: 'POST',
redirect: "follow", // manual, *follow, error
referrerPolicy: "no-referrer", 
mode: "cors", // no-cors, *cors, same-origin
cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
//      credentials: "same-origin", 
credentials: 'include',
headers: { "Content-Type": "application/json",
"ngrok-skip-browser-warning": true,
"Access-Control-Allow-Origin": "*"
},
      body: JSON.stringify({fn: document.querySelector('textarea').value.split('\n'),
                            documentContext: {},
      })
})
let json = await fn.json()
console.log(json)
window.location.hash = json
}

function MyDropdown() {
  return (
    <div className="text-right relative">
      <Menu as="div" className="relative inline-block text-left">
        <div>
          <Menu.Button className="mt-1 inline-flex w-full justify-center rounded-md bg-white px-4 py-2 text-sm font-medium hover:bg-opacity-30 focus:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75">
            Export
            <ChevronDownIcon
              className="ml-2 -mr-1 h-5 w-5 text-violet-200 hover:text-violet-100"
              aria-hidden="true"
            />
          </Menu.Button>
        </div>
        <Transition
          as={Fragment}
          enter="transition ease-out duration-100"
          enterFrom="transform opacity-0 scale-95"
          enterTo="transform opacity-100 scale-100"
          leave="transition ease-in duration-75"
          leaveFrom="transform opacity-100 scale-100"
          leaveTo="transform opacity-0 scale-95"
        >
          <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right divide-y divide-gray-100 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
            <div className="px-1 py-1 ">
              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active ? 'bg-violet-500 text-white' : 'text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                    onClick={exportToPdf}
                  >
                    {active ? (
                      <EditActiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    ) : (
                      <EditInactiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    )}
                    To PDF
                  </button>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active ? 'bg-violet-500 text-white' : 'text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                  >
                    {active ? (
                      <DuplicateActiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    ) : (
                      <DuplicateInactiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    )}
                    Fork to Custom URL
                  </button>
                )}
              </Menu.Item>
            </div>
            <div className="px-1 py-1">
              {/* <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active ? 'bg-violet-500 text-white' : 'text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                  >
                    {active ? (
                      <ArchiveActiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    ) : (
                      <ArchiveInactiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    )}
                    Fork
                  </button>
                )}
              </Menu.Item> */}
              {/* <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active ? 'bg-violet-500 text-white' : 'text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                  >
                    {active ? (
                      <MoveActiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    ) : (
                      <MoveInactiveIcon
                        className="mr-2 h-5 w-5"
                        aria-hidden="true"
                      />
                    )}
                    Move
                  </button>
                )}
              </Menu.Item> */}
            </div>
            {/* <div className="px-1 py-1">
              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active ? 'bg-violet-500 text-white' : 'text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                  >
                    {active ? (
                      <DeleteActiveIcon
                        className="mr-2 h-5 w-5 text-violet-400"
                        aria-hidden="true"
                      />
                    ) : (
                      <DeleteInactiveIcon
                        className="mr-2 h-5 w-5 text-violet-400"
                        aria-hidden="true"
                      />
                    )}
                    Delete
                  </button>
                )}
              </Menu.Item>
            </div> */}
          </Menu.Items>
        </Transition>
      </Menu>
    </div>
  )
}

function EditInactiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4 13V16H7L16 7L13 4L4 13Z"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
    </svg>
  )
}

function EditActiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4 13V16H7L16 7L13 4L4 13Z"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
    </svg>
  )
}

function DuplicateInactiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4 4H12V12H4V4Z"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
      <path
        d="M8 8H16V16H8V8Z"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
    </svg>
  )
}

function DuplicateActiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4 4H12V12H4V4Z"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
      <path
        d="M8 8H16V16H8V8Z"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
    </svg>
  )
}

function ArchiveInactiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        x="5"
        y="8"
        width="10"
        height="8"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
      <rect
        x="4"
        y="4"
        width="12"
        height="4"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
      <path d="M8 12H12" stroke="#A78BFA" strokeWidth="2" />
    </svg>
  )
}

function ArchiveActiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        x="5"
        y="8"
        width="10"
        height="8"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
      <rect
        x="4"
        y="4"
        width="12"
        height="4"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
      <path d="M8 12H12" stroke="#A78BFA" strokeWidth="2" />
    </svg>
  )
}

function MoveInactiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M10 4H16V10" stroke="#A78BFA" strokeWidth="2" />
      <path d="M16 4L8 12" stroke="#A78BFA" strokeWidth="2" />
      <path d="M8 6H4V16H14V12" stroke="#A78BFA" strokeWidth="2" />
    </svg>
  )
}

function MoveActiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M10 4H16V10" stroke="#C4B5FD" strokeWidth="2" />
      <path d="M16 4L8 12" stroke="#C4B5FD" strokeWidth="2" />
      <path d="M8 6H4V16H14V12" stroke="#C4B5FD" strokeWidth="2" />
    </svg>
  )
}

function DeleteInactiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        x="5"
        y="6"
        width="10"
        height="10"
        fill="#EDE9FE"
        stroke="#A78BFA"
        strokeWidth="2"
      />
      <path d="M3 6H17" stroke="#A78BFA" strokeWidth="2" />
      <path d="M8 6V4H12V6" stroke="#A78BFA" strokeWidth="2" />
    </svg>
  )
}

function DeleteActiveIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        x="5"
        y="6"
        width="10"
        height="10"
        fill="#8B5CF6"
        stroke="#C4B5FD"
        strokeWidth="2"
      />
      <path d="M3 6H17" stroke="#C4B5FD" strokeWidth="2" />
      <path d="M8 6V4H12V6" stroke="#C4B5FD" strokeWidth="2" />
    </svg>
  )
}


// function MyPopover() {
//   return (
//     <Popover className="relative">
//       <Popover.Button>About</Popover.Button>

//       <Popover.Panel className="absolute z-10 bg-white w-96 h-128 border border-purple-500">
// <div class="flex items-center justify-between">
//   <h3 class="text-base font-semibold text-sky-500">Get with all-access</h3>
//   <a class="inline-flex justify-center rounded-lg text-sm font-semibold py-2 px-3 bg-slate-900 text-white hover:bg-slate-700" href="/all-access">
//     <span>Get all-access <span aria-hidden="true">→</span></span>
//   </a>
//   </div>
//     <div class="mt-3 flex items-center">
//       <p class="mr-3 text-lg font-semibold text-slate-500 line-through">299</p>
//       <p class="text-[2.5rem] leading-none text-slate-900">$<span class="font-bold">50</span></p>
//       <p class="ml-3 text-sm"><span class="font-semibold text-slate-900">one-time payment</span>
//       <span class="text-slate-500">plus local taxes</span></p></div>
//       <p class="mt-6 text-sm leading-6 text-slate-600">Get lifetime access to all of the application UI, marketing, and ecommerce components, as well as all of our site templates for a single one-time purchase.</p>
//       <div class="relative mt-6 flex items-start rounded-xl border border-slate-600/10 bg-slate-50 p-3"><svg class="h-6 w-6" viewBox="0 0 24 24" fill="none"><path d="M19.4067 15.0682L18.9748 14.455C18.7377 14.6219 18.6181 14.9106 18.6677 15.1963L19.4067 15.0682ZM19.4071 8.93218L18.6682 8.8039C18.6186 9.08961 18.7382 9.37832 18.9752 9.54532L19.4071 8.93218ZM18.3643 5.6359L18.8946 5.10557L18.3643 5.6359ZM15.068 4.59307L14.4548 5.02495C14.6218 5.26205 14.9105 5.38162 15.1963 5.33201L15.068 4.59307ZM8.93194 4.59319L8.8037 5.33214C9.08943 5.38172 9.37814 5.26214 9.54513 5.02504L8.93194 4.59319ZM4.59298 8.93208L5.02488 9.54524C5.26194 9.37826 5.38151 9.08957 5.33193 8.80386L4.59298 8.93208ZM4.59286 15.0678L5.33181 15.1961C5.38141 14.9104 5.26185 14.6217 5.02478 14.4547L4.59286 15.0678ZM5.6357 18.3641L5.10537 18.8945L5.10537 18.8945L5.6357 18.3641ZM8.93204 19.407L9.54521 18.9751C9.37821 18.738 9.08949 18.6184 8.80375 18.668L8.93204 19.407ZM15.068 19.4069L15.1962 18.6679C14.9105 18.6184 14.6218 18.7379 14.4548 18.975L15.068 19.4069ZM8.46967 14.4697C8.17678 14.7626 8.17678 15.2374 8.46967 15.5303C8.76256 15.8232 9.23744 15.8232 9.53033 15.5303L8.46967 14.4697ZM15.5303 9.53033C15.8232 9.23744 15.8232 8.76256 15.5303 8.46967C15.2374 8.17678 14.7626 8.17678 14.4697 8.46967L15.5303 9.53033ZM9.75 9.75V9C9.33579 9 9 9.33579 9 9.75H9.75ZM9.7575 9.75H10.5075C10.5075 9.33579 10.1717 9 9.7575 9V9.75ZM9.7575 9.7575V10.5075C10.1717 10.5075 10.5075 10.1717 10.5075 9.7575H9.7575ZM9.75 9.7575H9C9 10.1717 9.33579 10.5075 9.75 10.5075V9.7575ZM14.25 14.25V13.5C13.8358 13.5 13.5 13.8358 13.5 14.25H14.25ZM14.2575 14.25H15.0075C15.0075 13.8358 14.6717 13.5 14.2575 13.5V14.25ZM14.2575 14.2575V15.0075C14.6717 15.0075 15.0075 14.6717 15.0075 14.2575H14.2575ZM14.25 14.2575H13.5C13.5 14.6717 13.8358 15.0075 14.25 15.0075V14.2575ZM20.25 12C20.25 13.014 19.7476 13.9108 18.9748 14.455L19.8385 15.6814C20.9931 14.8683 21.75 13.5226 21.75 12H20.25ZM18.9752 9.54532C19.7478 10.0896 20.25 10.9862 20.25 12H21.75C21.75 10.4777 20.9934 9.13219 19.8391 8.31903L18.9752 9.54532ZM17.834 6.16623C18.551 6.88326 18.8299 7.87262 18.6682 8.8039L20.1461 9.06045C20.3876 7.669 19.9713 6.18221 18.8946 5.10557L17.834 6.16623ZM15.1963 5.33201C16.1276 5.17034 17.1169 5.44919 17.834 6.16623L18.8946 5.10557C17.818 4.02892 16.3312 3.61256 14.9397 3.85412L15.1963 5.33201ZM12 3.75C13.0139 3.75 13.9106 4.25228 14.4548 5.02495L15.6812 4.16119C14.868 3.00672 13.5224 2.25 12 2.25V3.75ZM9.54513 5.02504C10.0893 4.25232 10.986 3.75 12 3.75V2.25C10.4775 2.25 9.13188 3.00678 8.31875 4.16133L9.54513 5.02504ZM6.1662 6.1664C6.88319 5.44941 7.87247 5.17054 8.8037 5.33214L9.06017 3.85423C7.66879 3.61278 6.18211 4.02916 5.10554 5.10574L6.1662 6.1664ZM5.33193 8.80386C5.17035 7.87264 5.44922 6.88338 6.1662 6.1664L5.10554 5.10574C4.02898 6.1823 3.61259 7.66895 3.85402 9.06031L5.33193 8.80386ZM3.75 12C3.75 10.9861 4.25224 10.0895 5.02488 9.54524L4.16108 8.31893C3.00667 9.13207 2.25 10.4776 2.25 12H3.75ZM5.02478 14.4547C4.2522 13.9104 3.75 13.0138 3.75 12H2.25C2.25 13.5223 3.00661 14.8678 4.16094 15.681L5.02478 14.4547ZM6.16603 17.8338C5.449 17.1168 5.17014 16.1274 5.33181 15.1961L3.85391 14.9395C3.61236 16.331 4.02873 17.8178 5.10537 18.8945L6.16603 17.8338ZM8.80375 18.668C7.87245 18.8297 6.88308 18.5508 6.16603 17.8338L5.10537 18.8945C6.18203 19.9711 7.66886 20.3875 9.06033 20.1459L8.80375 18.668ZM12 20.25C10.9861 20.25 10.0894 19.7477 9.54521 18.9751L8.31888 19.8389C9.13201 20.9933 10.4776 21.75 12 21.75V20.25ZM14.4548 18.975C13.9106 19.7477 13.0139 20.25 12 20.25V21.75C13.5224 21.75 14.8681 20.9932 15.6812 19.8387L14.4548 18.975ZM17.8334 17.8336C17.1165 18.5505 16.1273 18.8294 15.1962 18.6679L14.9398 20.1458C16.3311 20.3871 17.8176 19.9707 18.8941 18.8943L17.8334 17.8336ZM18.6677 15.1963C18.8292 16.1275 18.5503 17.1167 17.8334 17.8336L18.8941 18.8943C19.9705 17.8178 20.3869 16.3313 20.1456 14.94L18.6677 15.1963ZM9.53033 15.5303L15.5303 9.53033L14.4697 8.46967L8.46967 14.4697L9.53033 15.5303ZM9.75 10.5H9.7575V9H9.75V10.5ZM9.0075 9.75V9.7575H10.5075V9.75H9.0075ZM9.7575 9.0075H9.75V10.5075H9.7575V9.0075ZM10.5 9.7575V9.75H9V9.7575H10.5ZM9.375 9.75C9.375 9.54289 9.54289 9.375 9.75 9.375V10.875C10.3713 10.875 10.875 10.3713 10.875 9.75H9.375ZM9.75 9.375C9.95711 9.375 10.125 9.54289 10.125 9.75H8.625C8.625 10.3713 9.12868 10.875 9.75 10.875V9.375ZM10.125 9.75C10.125 9.95711 9.95711 10.125 9.75 10.125V8.625C9.12868 8.625 8.625 9.12868 8.625 9.75H10.125ZM9.75 10.125C9.54289 10.125 9.375 9.95711 9.375 9.75H10.875C10.875 9.12868 10.3713 8.625 9.75 8.625V10.125ZM14.25 15H14.2575V13.5H14.25V15ZM13.5075 14.25V14.2575H15.0075V14.25H13.5075ZM14.2575 13.5075H14.25V15.0075H14.2575V13.5075ZM15 14.2575V14.25H13.5V14.2575H15ZM13.875 14.25C13.875 14.0429 14.0429 13.875 14.25 13.875V15.375C14.8713 15.375 15.375 14.8713 15.375 14.25H13.875ZM14.25 13.875C14.4571 13.875 14.625 14.0429 14.625 14.25H13.125C13.125 14.8713 13.6287 15.375 14.25 15.375V13.875ZM14.625 14.25C14.625 14.4571 14.4571 14.625 14.25 14.625V13.125C13.6287 13.125 13.125 13.6287 13.125 14.25H14.625ZM14.25 14.625C14.0429 14.625 13.875 14.4571 13.875 14.25H15.375C15.375 13.6287 14.8713 13.125 14.25 13.125V14.625Z" fill="#94A3B8"></path></svg>
//       <p class="ml-2 text-sm leading-6 text-slate-700">
//         <strong class="font-semibold text-slate-900">Discounted</strong> 
//         — since you own other Tailwind UI products.</p></div>
//         <h4 class="sr-only">All-access features</h4>
//         <ul class="mt-10 space-y-8 border-t border-slate-900/10 pt-10 text-sm leading-6 text-slate-700">
//           {/* <li class="flex"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 32 32" class="h-8 w-8 flex-none"><path fill="#fff" d="M0 0h32v32H0z"></path><path fill="#E0F2FE" d="M23 22l7-4v7l-7 4v-7zM9 22l7-4v7l-7 4v-7zM16 11l7-4v7l-7 4v-7zM2 18l7 4v7l-7-4v-7zM9 7l7 4v7l-7-4V7zM16 18l7 4v7l-7-4v-7z"></path><path fill="#0EA5E9" d="M16 3l.372-.651a.75.75 0 00-.744 0L16 3zm7 4h.75a.75.75 0 00-.378-.651L23 7zM9 7l-.372-.651A.75.75 0 008.25 7H9zM2 18l-.372-.651A.75.75 0 001.25 18H2zm28 0h.75a.75.75 0 00-.378-.651L30 18zm0 7l.372.651A.75.75 0 0030.75 25H30zm-7 4l-.372.651a.75.75 0 00.744 0L23 29zM9 29l-.372.651a.75.75 0 00.744 0L9 29zm-7-4h-.75c0 .27.144.518.378.651L2 25zM15.628 3.651l7 4 .744-1.302-7-4-.744 1.302zm7 2.698l-7 4 .744 1.302 7-4-.744-1.302zm-6.256 4l-7-4-.744 1.302 7 4 .744-1.302zm-7-2.698l7-4-.744-1.302-7 4 .744 1.302zm-.744 7l7 4 .744-1.302-7-4-.744 1.302zm7 2.698l-7 4 .744 1.302 7-4-.744-1.302zm-6.256 4l-7-4-.744 1.302 7 4 .744-1.302zm-7-2.698l7-4-.744-1.302-7 4 .744 1.302zm20.256-4l7 4 .744-1.302-7-4-.744 1.302zm7 2.698l-7 4 .744 1.302 7-4-.744-1.302zm-6.256 4l-7-4-.744 1.302 7 4 .744-1.302zm-7-2.698l7-4-.744-1.302-7 4 .744 1.302zm13.256 5.698l-7 4 .744 1.302 7-4-.744-1.302zm-6.256 4l-7-4-.744 1.302 7 4 .744-1.302zM30.75 25v-7h-1.5v7h1.5zm-15.122-.651l-7 4 .744 1.302 7-4-.744-1.302zm-6.256 4l-7-4-.744 1.302 7 4 .744-1.302zM2.75 25v-7h-1.5v7h1.5zm14 0v-7h-1.5v7h1.5zM8.25 7v7h1.5V7h-1.5zm14 0v7h1.5V7h-1.5zm-7 4v7h1.5v-7h-1.5zm-7 11v7h1.5v-7h-1.5zm14 0v7h1.5v-7h-1.5z"></path></svg><p class="ml-5"><strong class="font-semibold text-slate-900">Over 500+ components</strong> <!-- -->— everything you need to build beautiful application UIs, marketing sites, ecommerce stores, and more.</p></li><li class="flex"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 32 32" class="h-8 w-8 flex-none"><path fill="#fff" d="M0 0h32v32H0z"></path><rect width="23" height="22" x="3" y="5" stroke="#0EA5E9" stroke-linejoin="round" stroke-width="1.5" rx="2"></rect><rect width="10" height="18" x="19" y="9" fill="#E0F2FE" stroke="#0EA5E9" stroke-linejoin="round" stroke-width="1.5" rx="2"></rect><circle cx="6" cy="8" r="1" fill="#0EA5E9"></circle><circle cx="9" cy="8" r="1" fill="#0EA5E9"></circle><path stroke="#0EA5E9" stroke-width="1.5" d="M3 11h16"></path></svg><p class="ml-5"><strong class="font-semibold text-slate-900">Every site template</strong> <!-- -->— beautifully designed, expertly crafted website templates built with modern technologies like React and Next.js.</p></li> */}
//         <li class="flex"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 32 32" class="h-8 w-8 flex-none"><path fill="#fff" d="M0 0h32v32H0z"></path><path fill="#E0F2FE" d="M13.168 18.828a4 4 0 110-5.656L15.997 16l1.5-1.5 1.328-1.328a4 4 0 110 5.656L15.996 16l-1.499 1.5-1.329 1.328z"></path><path stroke="#0EA5E9" stroke-linecap="round" stroke-width="1.5" d="M14.497 17.5l-1.329 1.328a4 4 0 110-5.656l5.657 5.656a4 4 0 100-5.656L17.496 14.5"></path><circle cx="16" cy="16" r="14" stroke="#0EA5E9" stroke-width="1.5"></circle></svg><p class="ml-5"><strong class="font-semibold text-slate-900">Lifetime access</strong> — get instant access to everything we have today, plus any new components and templates we add in the future.</p></li></ul><div class="relative mt-10 flex rounded-xl border border-slate-600/10 bg-slate-50 p-6"><svg fill="none" xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 flex-none" stroke="#94A3B8" stroke-width="1.5"><circle cx="11" cy="16" r="3.25" fill="#94A3B8" fill-opacity=".2"></circle><circle cx="21" cy="13" r="3.25" fill="#94A3B8" fill-opacity=".2"></circle><path d="M28.909 19c.223-.964.341-1.968.341-3 0-7.318-5.932-13.25-13.25-13.25S2.75 8.682 2.75 16c0 1.032.118 2.036.341 3" stroke-linecap="round"></path><path d="m18.031 29.016-2.187.109s-1.475-.118-1.827-.29c-1.049-.51-.579-2.915 0-3.95 1.157-2.064 3.752-5.135 7.125-5.135h.024c2.5 0 4.404 1.687 5.692 3.401-1.963 2.975-5.161 5.276-8.827 5.865Z" fill="#94A3B8" fill-opacity=".2" stroke-linejoin="round"></path><path d="m14.001 24.913.016-.027c.26-.465.593-.98.991-1.5-1.042-.918-2.374-1.636-3.988-1.636H11c-2.094 0-3.847 1.208-5.055 2.492a12.987 12.987 0 0 0 7.987 4.595l.057-.016c-1.004-.534-.555-2.868.012-3.908Z" fill="#94A3B8" fill-opacity=".2" stroke-linejoin="round"></path></svg>
//         <p class="ml-5 text-sm leading-6 text-slate-700"><strong class="font-semibold text-slate-900">Available for teams</strong> — get access to all of our components and templates plus any future updates for your entire team.</p>
//       </div>
//       </Popover.Panel>
//     </Popover>
//   )
// }

let world_festivals = [
  [
      "Amsterdam Dance Event 2023",
      "Amsterdam, Netherlands",
      "October 18-22, 2023 "
  ],
  [
      "NDK Festival 2023",
      "Caen, France",
      "October 18-26, 2023 "
  ],
  [
      "Pitchfork Paris 2023",
      "Paris, France",
      "November 6-12, 2023 "
  ],
  [
      "Pitchfork London 2023",
      "London, UK",
      "November 7-13, 2023"
  ],
  [
      "London Jazz Festival 2023",
      "London, UK",
      "November 10-19, 2023 "
  ],
  [
      "Rise Festival 2023",
      "Les 2 Alpes, France",
      "December 9-16, 2023 "
  ],
  [
      "Your Paradise Fiji 2023",
      "Mamanucas, Fiji",
      "December 10-16, 2023 "
  ],
  [
      "ADVERTISEMENT"
  ],
  [
      "Universo Paralello 2023",
      "Pratigi, Brazil",
      "December 27-January 3, 2024 "
  ],
  [
      "Woodford Folk Festival 2023",
      "Woodford, Australia",
      "December 27-January 1, 2024 "
  ],
  [
      "Labryinto 2024",
      "Guanacaste, Costa Rica",
      "December 27-January 2, 2024 "
  ],
  [
      "Rhythm and Vines 2023",
      "Gisborne, New Zealand",
      "December 29-31, 2023"
  ],
  [
      "The MusicFest Steamboat 2024",
      "Steamboat Springs, CO",
      "January 6-11, 2024 "
  ],
  [
      "Igloofest 2024",
      "Montreal, QC",
      "January 18-February 10, 2024 "
  ],
  [
      "Tamworth Country Music Festival 2024",
      "Tamworth, Australia ",
      "January 19-28, 2024 "
  ]
]

let festivals = [
  [
      "III Points 2023",
      "Miami, FL",
      "October 20-21, 2023"
  ],
  [
      "Boo! Seattle 2023",
      "Seattle, WA",
      "October 27-28, 2023"
  ],
  [
      "Escape Halloween 2023",
      "San Bernardino, CA",
      "October 27-28, 2023"
  ],
  [
      "Electric Daisy Carnival Orlando 2023",
      "Orlando, FL",
      "November 10-12, 2023"
  ],
  [
      "Disco Trip 2023",
      "Los Angeles, CA",
      "November 11, 2023"
  ],
  [
      "Dreamstate SoCal 2023",
      "Long Beach, CA",
      "November 17-18, 2023"
  ],
  [
      "Apocalypse: Zombieland 2023",
      "Long Beach, CA",
      "November 24-25, 2023"
  ],
  [
      "Bullseye Festival 2023",
      "San Bernardino, CA",
      "December 8-9, 2023"
  ],
  [
      "Forever Midnight Los Angeles 2023",
      "Los Angeles, CA",
      "December 30-31, 2023"
  ],
  [
      "Forever Midnight Las Vegas 2023",
      "Las Vegas, NV",
      "December 30-31, 2023"
  ],
  [
      "Dreamstate San Francisco 2023",
      "Long Beach, CA",
      "December 30, 2023"
  ],
  [
      "Together as One NYE 2023",
      "Tacoma, WA",
      "December 30-31, 2023 "
  ],
  [
      "Countdown NYE 2023",
      "San Bernardino, CA",
      "December 30-31, 2023"
  ],
  [
      "Beyond Wonderland 2024",
      "San Bernardino, CA",
      "March 22-23, 2024"
  ]
]

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

let cities = {
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
function generateData(amount) {
  var result = [];
  var data = {
    coin: "100",
    game_group: "GameGroup",
    game_name: "XPTO2",
    game_version: "25",
    machine: "20485861",
    vlt: "0"
  };
  for (var i = 0; i < amount; i += 1) {
    data.id = (i + 1).toString();
    result.push(Object.assign({}, data));
  }
  return result;
}



async function exportToPdf() {
//  ReactPDF.render(<MyPopover />, `example.pdf`);

  const doc = new jsPDF();

  doc.text("Remote Year - 52 cities in 52 weeks with overlapping conferences and so on.", 10, 10);
  let hello_world = Object.keys(cities)
  hello_world = Array.from(document.querySelectorAll('.city-selector option')).slice(1).map(_=>_.textContent)
  for (let i = 0; i < hello_world.length; i++) {
    let has_conf = ''
    let city_name = ''

  

    cool_beans.forEach(function (_, j) {
      let [conf_name, country, city, date] = _
      if (i === j) {
        has_conf = _.join(',')
        city_name = hello_world.find(_ => _.includes(city.replace(' ', '-'))) || hello_world[j]
      }
    });
    if (! city_name) continue
    let city_index =  Array.from(document.querySelectorAll('.city-selector option')).slice(1).map(_=>_.textContent)
    .indexOf(city_name)

    document.querySelector('.city-selector').value = hello_world[city_index]

    console.log('city_index', city_name, city_index)


    doc.addPage("a6", "l");
    doc.text(has_conf, 0, 20);
    doc.text(city_name, 0, 10);
    document.querySelector('table')
   let imgData =  Array.from(document.querySelectorAll('canvas'))
   imgData.forEach((canvas, index) => {
    let imgData = canvas.toDataURL();
    doc.addImage(imgData, "JPEG", 40, 40, 180, 180);
   })


      const svgString = document.querySelector("svg").innerHTML;
      // imgData =  "data:image/svg+xml;base64,"  + btoa(svgString);
      // doc.addImage(imgData, "JPEG", 15, 40, 180, 180);

      //const pngDataURL = canvas.toDataURL("image/png");

      function demoSvgDocument() {
        var doc = new jsPDF();
        var svgText =  document.querySelector("svg");
        var svgAsText = new XMLSerializer().serializeToString(svgText);
          doc.addSVG(svgAsText, 20, 20, doc.internal.pageSize.width - 20*2)
          // Save the PDF
          doc.save('TestSVG.pdf');
          window.doc = doc
    }    
    //demoSvgDocument()

    console.log('EXPORTING ', window.addToPdf())
    
    let labels = [
      "price",
      "longitude",
      "latitude",
      "h3_index",
      "avg_price",
      "good_deal",
      "commute_distance"
    ];



    const event = new Event('change', { bubbles: true });
    //document.querySelector('.city-selector').value = 'Winnipeg--Manitoba--Canada'
    document.querySelector('.city-selector').dispatchEvent(event);
    await sleep(100)
    let table = Array.from(document.querySelectorAll('tr')).map(_ => _.children[0].textContent + '           deal_score ' + _.children[2].textContent + '%')
    table.forEach((_, i) =>  { 
      //console.log("_", _)
      doc.text(_, 0, 75 + i * 5, ) 
      console.log(_)
    })
   

    //doc.table(1, 1, window.addToPdf().map(pair => Object.values(pair[1])), labels, { autoSize: true })
    //['url', 'deal_score', 'price', 'total_estimated_commute_time']

    
  }

  doc.save("a4.pdf");
}

function sleep(ms) {
  return new Promise(function (resolve) {
    resolve()
  }, ms)
}

function MyPopover() {
  return (
    <Popover className="relative font-serif">
      <Popover.Button>About</Popover.Button>

      <Popover.Panel className="absolute z-10 bg-white w-96 h-fit border border-purple-500 right-0">
<div class="h-fit w-fit">
<div>* Data Analyis in Natural Language</div>
<div>* Write code that is approved by 100% of people, not just 1% of programmers.</div>
<div>* Import/Export to SpreadSheet</div>
<div>* Automation for business and home</div>
<div>* Automatically Improve Decision Making and Resolve Ambiguity </div>
<div>* Crisis Diagnosis, Capacity Planning, Systems Design and Anomaly Detection</div>
<div>* A Structured Editor for Authoring Interactive & Data-Driven Articles</div>
<div>find research papers → identify problem → propose solution → prototype implementation → measure progress / results → propose improvements slowly and measure them</div>
Solution: New bills could be drafted in english and each sentence generates a UI component and server module that can be publicly viewed and utilized by everyone on earth. improvements to documents that create and execture infrastructure like "Snow Movers should vacumn the streets every day after snow at 8am" - their location should be viewable on a map, and any issues that occur will be logged in a database
</div>
      </Popover.Panel>
    </Popover>
  )
}

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)


  return (
    <header className="bg-white">
      <nav className="mx-auto flex full-w items-center justify-between p-6 lg:px-8" aria-label="Global">
        <div className="flex items-center gap-x-12">
            <span className="text-xl font-serif">Cooperation.Party</span>
          {/* <div className="hidden lg:flex lg:gap-x-12">
            {navigation.map((item) => (
              <a key={item.name} href={item.href} className="text-sm font-semibold leading-6 text-gray-900">
                {item.name}
              </a>
            ))}
          </div> */}
        </div>
        <div className="flex lg:hidden">
          <button
            type="button"
            className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
            onClick={() => setMobileMenuOpen(true)}
          >
            <span className="sr-only">Open main menu</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
        </div>
        <div className="hidden lg:flex">
          <a 
          onClick={fork}
          href="#" className="hidden rounded p-8 text-sm font-semibold leading-6 text-white bg-blue-500">
            Share <span aria-hidden="true">&rarr;</span>
          </a>
          {/* <a href="#about">About</a> */}
          <MyPopover />
          {/* <a href="#settings">Settings</a> */}
          <MyDropdown />
        </div>
      </nav>
      <Dialog as="div" className="lg:hidden" open={mobileMenuOpen} onClose={setMobileMenuOpen}>
        <div className="fixed inset-0 z-10" />
        <Dialog.Panel className="fixed inset-y-0 right-0 z-10 w-full overflow-y-auto bg-white px-6 py-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10">
          <div className="flex items-center justify-between">
            <a href="#" className="-m-1.5 p-1.5">
              <span className="sr-only">Your Company</span>
              <img
                className="h-8 w-auto"
                src="https://tailwindui.com/img/logos/mark.svg?color=indigo&shade=600"
                alt=""
              />
            </a>
            <button
              type="button"
              className="-m-2.5 rounded-md p-2.5 text-gray-700"
              onClick={() => setMobileMenuOpen(false)}
            >
              <span className="sr-only">Close menu</span>
              <XMarkIcon className="h-6 w-6" aria-hidden="true" />
            </button>
          </div>
          <div className="mt-6 flow-root">
            <div className="-my-6 divide-y divide-gray-500/10">
              <div className="space-y-2 py-6">
                {navigation.map((item) => (
                  <a
                    key={item.name}
                    href={item.href}
                    className="-mx-3 block rounded-lg px-3 py-2 text-base font-semibold leading-7 text-gray-900 hover:bg-gray-50"
                  >
                    {item.name}
                  </a>
                ))}
              </div>
              <div className="py-6">
                <a
                  href="#"
                  className="-mx-3 block rounded-lg px-3 py-2.5 text-base font-semibold leading-7 text-gray-900 hover:bg-gray-50"
                >
                  Log in
                </a>
              </div>
            </div>
          </div>
        </Dialog.Panel>
      </Dialog>
    </header>
  )
}