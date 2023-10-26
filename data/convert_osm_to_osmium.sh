#!/bin/bash

function run_cmd {
    echo "Starting: $1"
    eval "$1"
    if [ $? -eq 0 ]; then
        echo "Done"
    else
        echo "Error running the command"
    fi
}
export -f run_cmd

commands=(
'osmium extract --bbox=3.895168,51.370216,5.895168,53.370216 -o "/Volumes/T7\ Shield/osm/Amsterdam--North-Holland--The-Netherlands.osm.pbf" "/Volumes/T7\ Shield/osm/europe-latest.osm.pbf"
'osmium extract --bbox=99.516667,12.75,101.516667,14.75 -o "/Volumes/T7\ Shield/osm/Bangkok--Central-Thailand--Thailand.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=137.985688,-35.533333,139.985688,-33.533333 -o "/Volumes/T7\ Shield/osm/Barossa-Valley--South-Australia--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=142.833133,-39.261111,144.833133,-37.261111 -o "/Volumes/T7\ Shield/osm/Barwon-South-West--Vic--Victoria--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=115.407396,38.9042,117.407396,40.9042 -o "/Volumes/T7\ Shield/osm/Beijing--Beijing--China.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=-89.760092,16.497713,-87.760092,18.497713 -o "/Volumes/T7\ Shield/osm/Belize--Belize--Belize.osm.pbf" "/Volumes/T7\ Shield/osm/central-america-latest.osm.pbf"
'osmium extract --bbox=-59.417309,-35.611996,-57.417309,-33.611996 -o "/Volumes/T7\ Shield/osm/Buenos-Aires--Ciudad-Autónoma-de-Buenos-Aires--Argentina.osm.pbf" "/Volumes/T7\ Shield/osm/south-america-latest.osm.pbf"
'osmium extract --bbox=17.424055,-34.924869,19.424055,-32.924869 -o "/Volumes/T7\ Shield/osm/Cape-Town--Western-Cape--South-Africa.osm.pbf" "/Volumes/T7\ Shield/osm/africa-latest.osm.pbf"
'osmium extract --bbox=113.109497,21.396428,115.109497,23.396428 -o "/Volumes/T7\ Shield/osm/Hong-Kong--Hong-Kong--China.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=27.978359,40.008238,29.978359,42.008238 -o "/Volumes/T7\ Shield/osm/Istanbul--Marmara--Turkey.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=143.9631,-38.8136,145.9631,-36.8136 -o "/Volumes/T7\ Shield/osm/Melbourne--Victoria--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=151.5016,-32.7446,153.5016,-30.7446 -o "/Volumes/T7\ Shield/osm/Mid-North-Coast--New-South-Wales--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=143.9766,-39.2854,145.9766,-37.2854 -o "/Volumes/T7\ Shield/osm/Mornington-Peninsula--Victoria--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=152.295,-29.865,154.295,-27.865 -o "/Volumes/T7\ Shield/osm/Northern-Rivers--New-South-Wales--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=-44.172896,-23.906847,-42.172896,-21.906847 -o "/Volumes/T7\ Shield/osm/Rio-de-Janeiro--Rio-de-Janeiro--Brazil.osm.pbf" "/Volumes/T7\ Shield/osm/south-america-latest.osm.pbf"
'osmium extract --bbox=-71.64827,-34.47269,-69.64827,-32.47269 -o "/Volumes/T7\ Shield/osm/Santiago--Región-Metropolitana-de-Santiago--Chile.osm.pbf" "/Volumes/T7\ Shield/osm/south-america-latest.osm.pbf"
'osmium extract --bbox=120.473701,30.230416,122.473701,32.230416000000005 -o "/Volumes/T7\ Shield/osm/Shanghai--Shanghai--China.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=102.8198,0.3521000000000001,104.8198,2.3521 -o "/Volumes/T7\ Shield/osm/Singapore--Singapore--Singapore.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=150.2093,-34.8688,152.2093,-32.8688 -o "/Volumes/T7\ Shield/osm/Sydney--New-South-Wales--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=120.5654,24.033,122.5654,26.033 -o "/Volumes/T7\ Shield/osm/Taipei--Northern-Taiwan--Taiwan.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=145.6735,-42.4545,147.6735,-40.4545 -o "/Volumes/T7\ Shield/osm/Tasmania--Tasmania--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=138.6917,34.6895,140.6917,36.6895 -o "/Volumes/T7\ Shield/osm/Tokyo--Japan.osm.pbf" "/Volumes/T7\ Shield/osm/asia-latest.osm.pbf"
'osmium extract --bbox=-94.265,43.9778,-92.265,45.9778 -o "/Volumes/T7\ Shield/osm/Twin-Cities-MSA--Minnesota--United-States.osm.pbf" "/Volumes/T7\ Shield/osm/north-america-latest.osm.pbf"
'osmium extract --bbox=-124.1207,48.2827,-122.1207,50.2827 -o "/Volumes/T7\ Shield/osm/Vancouver--British-Columbia--Canada.osm.pbf" "/Volumes/T7\ Shield/osm/north-america-latest.osm.pbf"
'osmium extract --bbox=-124.3656,47.4284,-122.3656,49.4284 -o "/Volumes/T7\ Shield/osm/Victoria--British-Columbia--Canada.osm.pbf" "/Volumes/T7\ Shield/osm/north-america-latest.osm.pbf"
'osmium extract --bbox=-78.0369,37.9072,-76.0369,39.9072 -o "/Volumes/T7\ Shield/osm/Washington--D.C.--District-of-Columbia--United-States.osm.pbf" "/Volumes/T7\ Shield/osm/north-america-latest.osm.pbf"
'osmium extract --bbox=114.8589,-32.950500000000005,116.8589,-30.9505 -o "/Volumes/T7\ Shield/osm/Western-Australia--Western-Australia--Australia.osm.pbf" "/Volumes/T7\ Shield/osm/australia-oceania-latest.osm.pbf"
'osmium extract --bbox=-98.1384,48.8951,-96.1384,50.8951 -o "/Volumes/T7\ Shield/osm/Winnipeg--Manitoba--Canada.osm.pbf" "/Volumes/T7\ Shield/osm/north-america-latest.osm.pbf"
)

#printf "%s\n" "${commands[@]}" | xargs -I {} -P 16 bash -c 'run_cmd "{}"'


#printf "%s\n" "${commands[@]}" | parallel -j 10 --line-buffer
#printf "%s\n" "${commands[@]}" | xargs -I {} bash -c 'run_cmd "$@"' _ {}

#printf "%s\n" "${commands[@]}" | xargs -I CMD -P 10 bash -c "CMD"

for cmd in "${commands[@]}"; do
    run_cmd "$cmd"
done

# #!/bin/bash


# function run_cmd {
#     echo "Starting: $1"
#     eval "$1"
#     if [ $? -eq 0 ]; then
#         echo "Done"
#     else
#         echo "Error running the command"
#     fi
# }
# export -f run_cmd

# commands=(
# "osmium extract --bbox=3.895168,51.370216,5.895168,53.370216 -o /Volumes/T7 Shield/Amsterdam--North-Holland--The-Netherlands.osm.pbf /Volumes/T7 Shield/europe-latest.osm.pbf"
# "osmium extract --bbox=99.516667,12.75,101.516667,14.75 -o /Volumes/T7 Shield/Bangkok--Central-Thailand--Thailand.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=137.985688,-35.533333,139.985688,-33.533333 -o /Volumes/T7 Shield/Barossa-Valley--South-Australia--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=142.833133,-39.261111,144.833133,-37.261111 -o /Volumes/T7 Shield/Barwon-South-West--Vic--Victoria--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=115.407396,38.9042,117.407396,40.9042 -o /Volumes/T7 Shield/Beijing--Beijing--China.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=-89.760092,16.497713,-87.760092,18.497713 -o /Volumes/T7 Shield/Belize--Belize--Belize.osm.pbf /Volumes/T7 Shield/central-america-latest.osm.pbf"
# "osmium extract --bbox=-59.417309,-35.611996,-57.417309,-33.611996 -o /Volumes/T7 Shield/Buenos-Aires--Ciudad-Autónoma-de-Buenos-Aires--Argentina.osm.pbf /Volumes/T7 Shield/south-america-latest.osm.pbf"
# "osmium extract --bbox=17.424055,-34.924869,19.424055,-32.924869 -o /Volumes/T7 Shield/Cape-Town--Western-Cape--South-Africa.osm.pbf /Volumes/T7 Shield/africa-latest.osm.pbf"
# "osmium extract --bbox=113.109497,21.396428,115.109497,23.396428 -o /Volumes/T7 Shield/Hong-Kong--Hong-Kong--China.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=27.978359,40.008238,29.978359,42.008238 -o /Volumes/T7 Shield/Istanbul--Marmara--Turkey.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=143.9631,-38.8136,145.9631,-36.8136 -o /Volumes/T7 Shield/Melbourne--Victoria--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=151.5016,-32.7446,153.5016,-30.7446 -o /Volumes/T7 Shield/Mid-North-Coast--New-South-Wales--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=143.9766,-39.2854,145.9766,-37.2854 -o /Volumes/T7 Shield/Mornington-Peninsula--Victoria--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=152.295,-29.865,154.295,-27.865 -o /Volumes/T7 Shield/Northern-Rivers--New-South-Wales--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=-44.172896,-23.906847,-42.172896,-21.906847 -o /Volumes/T7 Shield/Rio-de-Janeiro--Rio-de-Janeiro--Brazil.osm.pbf /Volumes/T7 Shield/south-america-latest.osm.pbf"
# "osmium extract --bbox=-71.64827,-34.47269,-69.64827,-32.47269 -o /Volumes/T7 Shield/Santiago--Región-Metropolitana-de-Santiago--Chile.osm.pbf /Volumes/T7 Shield/south-america-latest.osm.pbf"
# "osmium extract --bbox=120.473701,30.230416,122.473701,32.230416000000005 -o /Volumes/T7 Shield/Shanghai--Shanghai--China.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=102.8198,0.3521000000000001,104.8198,2.3521 -o /Volumes/T7 Shield/Singapore--Singapore--Singapore.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=150.2093,-34.8688,152.2093,-32.8688 -o /Volumes/T7 Shield/Sydney--New-South-Wales--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=120.5654,24.033,122.5654,26.033 -o /Volumes/T7 Shield/Taipei--Northern-Taiwan--Taiwan.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=145.6735,-42.4545,147.6735,-40.4545 -o /Volumes/T7 Shield/Tasmania--Tasmania--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=138.6917,34.6895,140.6917,36.6895 -o /Volumes/T7 Shield/Tokyo--Japan.osm.pbf /Volumes/T7 Shield/asia-latest.osm.pbf"
# "osmium extract --bbox=-94.265,43.9778,-92.265,45.9778 -o /Volumes/T7 Shield/Twin-Cities-MSA--Minnesota--United-States.osm.pbf /Volumes/T7 Shield/north-america-latest.osm.pbf"
# "osmium extract --bbox=-124.1207,48.2827,-122.1207,50.2827 -o /Volumes/T7 Shield/Vancouver--British-Columbia--Canada.osm.pbf /Volumes/T7 Shield/north-america-latest.osm.pbf"
# "osmium extract --bbox=-124.3656,47.4284,-122.3656,49.4284 -o /Volumes/T7 Shield/Victoria--British-Columbia--Canada.osm.pbf /Volumes/T7 Shield/north-america-latest.osm.pbf"
# "osmium extract --bbox=-78.0369,37.9072,-76.0369,39.9072 -o /Volumes/T7 Shield/Washington--D.C.--District-of-Columbia--United-States.osm.pbf /Volumes/T7 Shield/north-america-latest.osm.pbf"
# "osmium extract --bbox=114.8589,-32.950500000000005,116.8589,-30.9505 -o /Volumes/T7 Shield/Western-Australia--Western-Australia--Australia.osm.pbf /Volumes/T7 Shield/australia-oceania-latest.osm.pbf"
# "osmium extract --bbox=-98.1384,48.8951,-96.1384,50.8951 -o /Volumes/T7 Shield/Winnipeg--Manitoba--Canada.osm.pbf /Volumes/T7 Shield/north-america-latest.osm.pbf"
# )


# #printf "%s\n" "${commands[@]}" | xargs -I {} bash -c 'run_cmd "$@"' _ {}
# printf "%s\n" "${commands[@]}" | xargs -I CMD -P 10 bash -c "CMD"

    
# #printf "%s\n" "${commands[@]}" | parallel -j 10 --line-buffer
