#!/bin/bash
# Retrain TF object detection model.
# Ref: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md#running-locally
# Copyright (c) 2019 Lindo St. Angel.

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Starts retraining detection model.

  --pipeline_config_path - Path to pipeline config file (default ,/configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config)
  --train_dir - Path to train directory (default ./train)
  --num_training_steps - Number of training steps to run (default 1400)
  --sample_1_of_n_eval_examples - Will sample one of every n eval input examples (default 1).
  --help - Display this help.
END_OF_USAGE
}

# Defaults - will get overridden if provided on cmd line.
pipeline_config_path=./configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config
train_dir=./train
num_training_steps=1400
sample_1_of_n_eval_examples=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline_config_path)
      pipeline_config_path=$2
      shift 2 ;;
    --train_dir)
      train_dir=$2
      shift 2 ;;
    --num_training_steps)
      num_training_steps=$2
      shift 2 ;;
    --sample_1_of_n_eval_examples)
      sample_1_of_n_eval_examples=$2
      shift 2 ;;    
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

python3 research/object_detection/model_main.py \
  --pipeline_config_path="${pipeline_config_path}" \
  --model_dir="${train_dir}" \
  --num_train_steps="${num_training_steps}" \
  --sample_1_of_n_eval_examples="${sample_1_of_n_eval_examples}" \
  --alsologtostderr