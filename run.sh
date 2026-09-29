#!/bin/bash
# submits a job and shows its log live, starting fresh each time
# usage: ./run.sh masking_reuters

cd "$(dirname "$0")"
rm -f "$1.log"
sbatch "$1.job" && tail -F "$1.log"
