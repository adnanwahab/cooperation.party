//f(apt) -> list of googlemap
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path')

let cache = {};
async function getImgUrl(aptListing, idx) {
    let origAptListing = aptListing;

    function urlToFileName(url) {
        return url.match(/rooms\/(\d+)/)[1]
      }
      aptListing = urlToFileName(aptListing);

    let url = path.resolve(`data/airbnb/gm/${aptListing}.json`);
    console.log(url, aptListing)
    //console.log(aptListing, idx);
    //if (aptListing in cache) {return}
    cache[aptListing] = true;
    console.log(fs.existsSync(url), url);
    
    if (fs.existsSync(`data/airbnb/gm/${aptListing}.json`)) return console.log('cached url')
    //if (fs.existsSync(url)) return

    const browser = await puppeteer.launch({ headless: true }); // Replace true with false if you want to see the browser
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.goto(origAptListing);

    await page.waitForSelector('section');
    await page.waitForSelector('.hnwb2pb.dir.dir-ltr')
      delay(3000)
    await page.evaluate(() => {
        //window.scrollBy(0, 100000);
        setTimeout(function () {
            document.querySelectorAll('.hnwb2pb.dir.dir-ltr')[3].scrollIntoView()

        }, 1000)
    });

    await page.waitForSelector('.gm-style img');
    delay(3000)
    let imgUrls = await page.$$eval('.gm-style img', imgs => imgs.map(img => img.src).filter(src => src.indexOf("maps.googleapis.com") !== -1));
    if (imgUrls.length == 0) {
        delay(3000)
        imgUrls = await page.$$eval('.gm-style img', imgs => imgs.map(img => img.src).filter(src => src.indexOf("maps.googleapis.com") !== -1));
    }
    if (imgUrls.length === 0) { return console.log('could not render google maps')}
    console.log('saving to ' + url);
    fs.writeFileSync(url, JSON.stringify(imgUrls.slice(0, 6)));

    await browser.close();

    return imgUrls;
}

// Helper function to delay the script execution for `ms` milliseconds
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

(async () => {
    const urls = JSON.parse(fs.readFileSync(process.argv[2]))
    console.log(urls, process.argv[2])
    for (let idx = 0; idx < urls.length; idx++) {
        const url = urls[idx];
        const task = getImgUrl(url['link'], idx); // Replace 'link' with the appropriate key for your URLs
        await task; // If you wish to run these tasks concurrently, you can collect them in an array and use `Promise.all()`
    }
})();