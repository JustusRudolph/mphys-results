#!/bin/bash
CHECK_SPECIFIC_MODULE_ONLY=0
OUT_DIR="mphys-results/outputs/experiment/"
N_RUNS=5
OPTIND=1  # reset before while loop
while getopts "d:n:h" opt
do
  case "$opt" in
    d)
      echo "Setting output directory to $OPTARG"
      OUT_DIR=$OPTARG
      ;;
    n)
      N_RUNS=$OPTARG
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

for i in `seq 1 $N_RUNS`
do
  OUT_PATH=$OUT_DIR$i".txt"
  build/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_pixels/ --events=1 --run_cpu=1 &> $OUT_PATH
done

echo "Finished simulation."

# reset for next run
unset CHECK_SPECIFIC_MODULE
unset OUT_PATH
