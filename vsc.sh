#!/bin/bash
# sync code to VSC, submit a job, stream its log live, pull results back when it ends
# usage: ./vsc.sh masking

set -e
cd "$(dirname "$0")"

JOB="$1"
HOST=vsc5
REMOTE=/gpfs/data/fs71186/baumann/bsc

if [ -z "$JOB" ] || [ ! -f "jobs/$JOB.job" ]; then
  echo "usage: ./vsc.sh <jobname>   (needs jobs/<jobname>.job)"
  exit 1
fi

# 1. push code
./sync_vsc.sh

# 2. submit, report queue status while pending, stream the log until the job leaves the queue
ssh -t "$HOST" "
  cd $REMOTE
  rm -f $JOB.log
  JID=\$(sbatch --parsable jobs/$JOB.job) || exit 1
  echo \"submitted job \$JID\"
  squeue --start -j \$JID
  # until the log appears: a status line every 60s, and immediately when the state changes
  LAST_STATE=
  LAST_PRINT=0
  while [ ! -f $JOB.log ]; do
    LINE=\$(squeue -h -j \$JID -o '%T|%r|%S' 2>/dev/null)
    [ -n \"\$LINE\" ] || break
    STATE=\$(echo \"\$LINE\" | cut -d'|' -f1)
    REASON=\$(echo \"\$LINE\" | cut -d'|' -f2)
    START=\$(echo \"\$LINE\" | cut -d'|' -f3)
    NOW=\$(date +%s)
    if [ \"\$STATE\" != \"\$LAST_STATE\" ] || [ \$((NOW - LAST_PRINT)) -ge 60 ]; then
      echo \"\$(date +%H:%M:%S)  state \$STATE  reason \$REASON  est. start \$START\"
      LAST_STATE=\$STATE
      LAST_PRINT=\$NOW
    fi
    sleep 5
  done
  if [ ! -f $JOB.log ]; then
    echo
    echo \"job \$JID left the queue without writing $JOB.log (failed before it started, or was cancelled)\"
    sacct -j \$JID --format=JobID,State,Elapsed,ExitCode -X
    exit 0
  fi
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
