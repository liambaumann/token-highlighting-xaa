#!/bin/bash
# sync code to VSC, submit a job, stream its log live, pull results back when it ends
# usage: ./vsc.sh masking

set -e
cd "$(dirname "$0")"

JOB="$1"
HOST=vsc5
REMOTE=/gpfs/data/fs71186/baumann/bsc

if [ -z "$JOB" ] || [ ! -f "$JOB.job" ]; then
  echo "usage: ./vsc.sh <jobname>   (needs <jobname>.job in this folder)"
  exit 1
fi

# 1. push code
./sync_vsc.sh

# 2. submit, stream the log until the job leaves the queue
ssh -t "$HOST" "
  cd $REMOTE
  rm -f $JOB.log
  JID=\$(sbatch --parsable $JOB.job) || exit 1
  echo \"submitted job \$JID\"
  while [ ! -f $JOB.log ]; do
    squeue -h -j \$JID | grep -q . || break
    sleep 2
  done
  tail -n +1 -F $JOB.log 2>/dev/null &
  TAIL=\$!
  while squeue -h -j \$JID | grep -q .; do sleep 5; done
  sleep 2
  kill \$TAIL 2>/dev/null
  echo
  sacct -j \$JID --format=JobID,State,Elapsed,ExitCode -X
"

# 3. pull results and the log back
rsync -avzh "$HOST:$REMOTE/results/" ./results/
rsync -avzh "$HOST:$REMOTE/$JOB.log" ./outputs/ 2>/dev/null || true
echo "results pulled to ./results"
