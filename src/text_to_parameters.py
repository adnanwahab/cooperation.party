from .main import getAirbnbs
import subprocess

#getAirbnbs()


def get_all_airbnbs():
    from ipynb.fs.defs.geospatial import getCityList

    for location in getCityList():
       for i in location:
        args = [
            "node",
            "airbnb_get_img_url.js",
            f'{location}_apt.json'
            #'Kuala\ Lumpur--Malaysia_apt.json'
        ]
       completed_process = subprocess.run(args
                                       #, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                                       )
get_all_airbnbs()