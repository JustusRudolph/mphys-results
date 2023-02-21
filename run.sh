#!/bin/bash
CHECK_SPECIFIC_MODULE_ONLY=0
OUT_PATH="mphys-results/outputs/out.txt"
OPTIND=1  # reset before while loop
while getopts "sp:h" opt
do
  case "$opt" in
    s)
      echo "Only one module to look at."
      CHECK_SPECIFIC_MODULE_ONLY=1
      ;;
    p)
      echo "Setting path to $OPTARG"
      OUT_PATH=$OPTARG
      ;;
   
    ?|h)
      echo "To use analyser: $0 $1 [-s] [-p <path>]"
      echo "-s: (single_cell) sets the analysis to only consider a single module, the number of which is recognised by the programme."
      echo "-p: Requires an argument. Set the output path for all the piped printf messages from the simulation."
      return  # when sourcing, don't use exit
      ;;
  esac
done
echo "Starting reconstruction simulation..."
build/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_pixels/ --events=1 --run_cpu=1 &> $OUT_PATH


echo "Finished simulation."
echo "Starting analysing..."

python3 mphys-results/activation_analysis.py -p $OUT_PATH -s $CHECK_SPECIFIC_MODULE_ONLY


# reset for next run
unset CHECK_SPECIFIC_MODULE
unset OUT_PATH
