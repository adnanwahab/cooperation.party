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




#printf "%s\n" "${commands[@]}" | xargs -I {} bash -c 'run_cmd "$@"' _ {}
printf "%s\n" "${commands[@]}" | xargs -I CMD -P 10 bash -c "CMD"

    
#printf "%s\n" "${commands[@]}" | parallel -j 10 --line-buffer
