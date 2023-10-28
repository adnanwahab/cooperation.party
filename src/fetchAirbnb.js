const puppeteer = require('puppeteer');
const fs = require('fs')
const path = require('path')
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const get_img_url = async (apt_listing) => {
    //const browser = await puppeteer.launch();
    //console.log(apt_listing)
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    await page.setViewport({
        width: 1920,
        height: 1080,
      });
    // Go to the Twitter homepage
    await page.goto(apt_listing);
    // Wait for the page to load completely
    //await autoScroll(page);
    // await page.screenshot({
    //     path: 'yoursite.png',
    //     fullPage: true
    // });
    let query = 'section'
    await page.waitForSelector(query);
    page.evaluate(_ => {
        //document.documentElement.scrollHeight
        window.scrollBy(0, 100000);
      });
    await sleep(3000)
    page.evaluate(_ => {
        //document.documentElement.scrollHeight
        window.scrollBy(0, 100000);

        document.querySelectorAll('.hnwb2pb.dir.dir-ltr')[3].scrollIntoView()
        document.querySelector('.cj0q2ib.sne7mb7.rp6dtyx.c1y4i074.dir.dir-ltr').click()
      });
      await sleep(1000)
    //console.log('SCROLLING IS DONE')
    // Get all the tweets on the page
    let selector = ('.gm-style img')
    const tweets = await page.$$eval(selector, (img) => {
        //console.log(img)
        return img.map(_ => _.src).filter(_ => _.indexOf('maps.googleapis.com') !== -1)
    });


    fs.writeFileSync('airbnb_map.json', JSON.stringify(tweets));
    //console.log(tweets)
    await browser.close();
}

function urlToFileName(url) {
    const match = url.match(/rooms\/(\d+)/);
    if (match && match[1]) {
        return match[1];
    }
    // Handle the case where there's no match.
    // You can either return a default value, throw a custom error, or simply return null.
    return null;  // or throw new Error("Invalid URL format");
}


const get_apt = async (city_name, page, latlng) => {
    let url = `https://www.airbnb.com/s/${city_name}/homes`
    const fp = path.resolve(`data/airbnb/apt/${city_name}.json`);
    await page.goto(url, { waitUntil: 'networkidle2' });
    const qs = '.cy5jw6o.dir.dir-ltr a';
    await page.waitForSelector(qs);
    let tweets = await page.$$eval(qs, (tweetNodes) => {
        return tweetNodes.map(tweet => (tweet.href) );
    })
    
    tweets = Array.from(new Set(tweets.map(urlToFileName).filter(_ => _)))

    let result = {}
    tweets.forEach((listing_id) =>  {
        let newlatlng = [latlng[0] + Math.random() * .1, latlng[1] + Math.random() * .1]
        result[listing_id] = newlatlng
    })
    console.log(city_name, tweets.length, latlng)
    fs.writeFileSync(fp, JSON.stringify(result, null, 2));
}

async function main() {
    const browser = await puppeteer.launch({ headless: true });  // Change to false if you want to view the browser
    const page = await browser.newPage();
    let locations = JSON.parse(fs.readFileSync('data/all_city_names.json'))
    for (let city_name in locations) {
        await get_apt(city_name, page, locations[city_name])
        await sleep(500)
    }
    await browser.close();
}

main()



const makeURL = ({ne_lng, ne_lat, sw_lat, sw_lng, city_name, zoom_level}) => `https://www.airbnb.com/s/${city_name}/homes?place_id=ChIJ674hC6Y_WBQRujtC6Jay33k&refinement_paths%5B%5D=%2Fhomes&flexible_trip_dates%5B%5D=april&flexible_trip_dates%5B%5D=august&flexible_trip_dates%5B%5D=december&flexible_trip_dates%5B%5D=february&flexible_trip_dates%5B%5D=january&flexible_trip_dates%5B%5D=july&flexible_trip_dates%5B%5D=june&flexible_trip_dates%5B%5D=march&flexible_trip_dates%5B%5D=may&flexible_trip_dates%5B%5D=november&flexible_trip_dates%5B%5D=october&flexible_trip_dates%5B%5D=september&flexible_trip_lengths%5B%5D=one_week&date_picker_type=flexible_dates&search_type=user_map_move&tab_id=home_tab&query=cairo&monthly_start_date=2023-10-01&monthly_length=3&price_filter_input_type=0&price_filter_num_nights=5&channel=EXPLORE&ne_lat=${ne_lat}&ne_lng=${ne_lng}&sw_lat=${sw_lat}&sw_lng=${sw_lng}&zoom=16&zoom_level=16&search_by_map=true`
const fetch100Pages = (location) => {
    let coord = geo_coords[location]
    let BB = [[coord[0] - 2, coord[1] - 2], [coord[1] + 2, coord[1] + 2]]
    for (let i = 0; i < 5; i++ ) {
        for (let j = 0; j < 5; j++ ) {
            let bb = BB.slice()
            let ne_lng = bb[0][0] + i * .1
            let ne_lat = bb[0][1] + j * .1
            let sw_lng = bb[1][0] + i * .1
            let sw_lat = bb[1][1] + j * .1
            let params = {
                ne_lng, ne_lat, sw_lng, sw_lat, zoom_level: 16
            }
            const url = makeURL(params)
            console.log(url)
            console.log(params)
            setTimeout(function () {
                getApt(url, location, i * j + i)
            }, 3000 * i * j + i)
        }
    }
}