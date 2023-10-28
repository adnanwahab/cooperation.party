const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path')
async function sleep (ms) { 
    console.log('sleeping for ' + ms)
    return new Promise((resolve) => {
        setTimeout(() => resolve(), ms)
    })
}
async function getApt(location, page) {
    const fp = path.resolve(`data/airbnb/apt/${location}.json`);
    const url = `https://www.airbnb.com/s/${location}/homes/`
    await page.goto(url);
    await sleep(1000)
    const qs = '.cy5jw6o.dir.dir-ltr a';
    let tweets = [];
    function urlToFileName(url) {
        return url
        return 'https://www.airbnb.com/rooms/' + url.match(/rooms\/(\d+)/)[1]
      }
    try {
        await page.waitForSelector(qs);
        tweets = await page.$$eval(qs, (tweetNodes) => {
            return tweetNodes.map(tweet => (tweet.href) );
        })
    } catch (error) {
        console.error('Error:', error);
        return [];
    }
    
    console.log(`writing ${location}.json`)
  
    let apt = await fs.readFile(fp);
    tweets = Array.from(new Set(tweets.map(urlToFileName).concat(apt)))
    console.log(apt.length, tweets.length)
    await fs.writeFile(fp, JSON.stringify(tweets, null, 2));
    console.log(location, 'GET apt')
    return tweets;
}
// Example usage



async function main() {
    const browser = await puppeteer.launch({ headless: false });  // Change to false if you want to view the browser
    const page = await browser.newPage();
    let locations = JSON.parse(fs.readFileSync('data/all_city_names.json'))
    for (let city_name in locations) {
        getApt(city_name, page)
    }
    await browser.close();
}

main()

//let location = process.argv[2]
//location = location.replace(', ','--')
//fetch100Pages(location)