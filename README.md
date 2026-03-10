## Instructions
1. Download the Tweets csvs and experiment config YAML file.
2. Extract an csvs and place them in desired location. Update the yaml file to point to these files.
3. Run the following to perform the experiment:
```
python3 main.py --config=path/to/config.yaml --test-full-test-set
```
4. To format the metrics to use the confusion matrix, use the following:
```
python3 scripts/calc_f1_scores.py tb_logs/classifier_<timestamp>_crossval.csv --output crossval.csv
```
for each of crossval, training_agreed, and training_disagreed.

## Original Experiment Setup
### Common Configuration
CPU: 5955 WX Threadripper
Motherboard: ASUS WRX80
RAM: 256GB
OS: Ubuntu 25.04 (GNU/Linux 6.14.0-37-generic x86_64)
NVIDIA Drivers: 580.95.05
CUDA Version: 13.0
Python: 3.13.3 

### BERT Base
GPU: 1x RTX 3090

### BERT Large
GPU: 2x RTX 3090

### Notes
Although with the code, consecutive runs will be identical, there seem to be differences in results when this code is put on different systems. One cause of this is the scheduling of operations on the GPUs, which can cause rounding discrepancies, leading to errors propagating. There may be other slight differences in the system, but these could not be uncovered. Running several times, the results show the same general trend.

There is also a difference between the accuracy and F1-score metrics output by Pytorch and calculating using the confusion matrix gives different numbers. This is the purpose of running the second script to calculate the accuracy and F1 scores.
