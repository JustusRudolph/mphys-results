#!/bin/bash
CHECK_SPECIFIC_MODULE_ONLY=0
EXP_DIR="mphys-results/outputs/experiment/"
N_RUNS=5
N_EVENTS=1
OPTIND=1  # reset before while loop
while getopts "d:n:e:h" opt
do
  case "$opt" in
    d)
      echo "Setting output directory to $OPTARG"
      EXP_DIR=$OPTARG
      ;;
    n)
      N_RUNS=$OPTARG
      ;;
    e)
      N_EVENTS=$OPTARG
      ;;
    ?|h)
      echo "To use analyser: $0 $1 [-s] [-p <path>]"
      echo "-s: (single_cell) sets the analysis to only consider a single module, the number of which is recognised by the programme."
      echo "-p: Requires an argument. Set the output path for all the piped printf messages from the simulation."
      return  # when sourcing, don't use exit
      ;;
  esac
done

# create experiment directory
if [ -d $EXP_DIR ] 
  then
    echo "Directory $EXP_DIR already exists." 

  else
    mkdir $EXP_DIR
fi

# create directories for the thread numbers
for i in `seq 320 64 1024`
do
  OUT_PATH=$EXP_DIR$i
  if [ -d $OUT_PATH ] 
  then
    echo "Directory $OUT_PATH already exists." 

  else
    mkdir $OUT_PATH
  fi
  old_iter=$(($i-64))
  # echo $old_iter
  # echo $i
  sed -i "400s/$old_iter/$i/" /home/justusrudolph/Documents/University/Year5/Project/traccc_proj/traccc/device/cuda/src/clusterization/clusterization_algorithm.cu
  echo
  echo Compiling for $i threads...
  cmake --build build/ > /dev/null 2>&1 # build with current and pipe into nothing
  
  source mphys-results/experiment_run.sh -n $N_RUNS -d $OUT_PATH/ -e $N_EVENTS

done
# reset back to 64
sed -i "400s/1024/64/" /home/justusrudolph/Documents/University/Year5/Project/traccc_proj/traccc/device/cuda/src/clusterization/clusterization_algorithm.cu

echo "Finished running experiment."

# reset for next run
unset EXP_DIR
unset OUT_PATH
unset N_RUNS
unset N_EVENTS