#!/bin/bash
CHECK_SPECIFIC_MODULE_ONLY=0
OUT_DIR="mphys-results/outputs/experiment/"
N_EVENTS=1
N_RUNS=5
OPTIND=1  # reset before while loop
while getopts "d:n:e:h" opt
do
  case "$opt" in
    d)
      echo "Setting output directory to $OPTARG"
      OUT_DIR=$OPTARG
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
echo "Starting reconstruction simulation experiment..."

for j in `seq 1 $N_RUNS`
do
  OUT_PATH=$OUT_DIR$j".txt"
  build/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_pixels/ --events=$N_EVENTS --run_cpu=1 &> $OUT_PATH
done

echo "Finished simulation."

# reset for next run
unset OUT_DIR
unset OUT_PATH
unset j
