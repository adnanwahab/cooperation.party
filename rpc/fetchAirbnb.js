const puppeteer = require('puppeteer');
const fs = require('fs')


//given a city
//get all apt urls get 20

//given an apt url
//get the img urls

//make these callable from python
let fn = process.argv[2]
async function delay(ms) {
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
    await delay(3000)
    page.evaluate(_ => {
        //document.documentElement.scrollHeight
        window.scrollBy(0, 100000);

        document.querySelectorAll('.hnwb2pb.dir.dir-ltr')[3].scrollIntoView()
        document.querySelector('.cj0q2ib.sne7mb7.rp6dtyx.c1y4i074.dir.dir-ltr').click()
      });
      await delay(1000)
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





const get_apt = async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    // Go to the Twitter homepage
    let url = 'https://www.airbnb.com/s/New-Delhi--NCT-of-Delhi--India/homes?tab_id=home_tab&refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2023-10-01&monthly_length=3&price_filter_input_type=0&price_filter_num_nights=5&channel=EXPLORE&query=New%20Delhi%2C%20NCT%20of%20Delhi%2C%20India&date_picker_type=calendar&source=structured_search_input_header&search_type=filter_change&place_id=ChIJLbZ-NFv9DDkRzk0gTkm3wlI&pagination_search=true&cursor=eyJzZWN0aW9uX29mZnNldCI6MCwiaXRlbXNfb2Zmc2V0IjowLCJ2ZXJzaW9uIjoxfQ%3D%3D'
    await page.goto(url, { waitUntil: 'networkidle2' });
    
    // Wait for the page to load completely
    await page.waitForSelector('.timeline');
    
    // Get all the tweets on the page
    const tweets = await page.$$eval('.tweet', (tweets) => {
        return tweets.map((tweet) => ({
            text: tweet.querySelector('.text').innerText,
            author: tweet.querySelector('.author').innerText,
            date: tweet.querySelector('.date').innerText,
        }));
    });
    
    console.log(tweets);
    
    await browser.close();
}

if (fn === 'get_apt') get_apt(process.argv[3])
if (fn === 'get_img_url') get_img_url(process.argv[3])