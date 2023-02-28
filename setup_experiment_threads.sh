#!/bin/bash
CHECK_SPECIFIC_MODULE_ONLY=0
EXP_DIR="mphys-results/outputs/experiment/"
OPTIND=1  # reset before while loop
while getopts "d:h" opt
do
  case "$opt" in
    d)
      echo "Setting output directory to $OPTARG"
      EXP_DIR=$OPTARG
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
for i in `seq 64 64 1024`
do
  OUT_PATH=$EXP_DIR$i
  if [ -d $OUT_PATH ] 
  then
    echo "Directory $OUT_PATH already exists." 

  else
    mkdir $OUT_PATH
  fi
  
done

echo "Finished setting up experiment."

# reset for next run
unset EXP_DIR
unset OUT_PATH
