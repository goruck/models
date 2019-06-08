# TensorFlow Models with Edge TPU Training

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org) as well as addtional code added to train object detection models for the edge tpu:

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

Edge TPU training for the object detection models in the research folder can be done by the following steps.

1. Clone ```https://github.com/goruck/models/tree/edge-tpu```.

2. Set up environment

```bash
$ cd <path_to_your_tensorflow_installation>/models/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

You should set ```PYTHONPATH``` in ```~/.bashrc``` so it takes effect when you open a new shell. 

3. Collect images and label

I used the program [view-mongo-images.py](https://github.com/goruck/smart-zoneminder/blob/master/face-det-rec/view-mongo-images.py) to collect jpeg images of my family that were detected by a resnet101 object detector and identified by a dlib-based face recognizer. This detector / recognizer is very accurate (but expensive). The program can also save image detection labels in Pascal VOC format that was used when I manually determined the detection was correct. In the cases it was not I used [labelImg](https://github.com/tzutalin/labelImg) to label the images myself, again in the Pascal VOC format. The images and labels are stored in the ```images``` and ```annotations/xml``` folders, respectively. 

About 200 images per class (or family member in this case) is sufficient to re-train most models in my experience.

4. Create Label Map (.pbtxt)

Classes need to be listed in the label map. Since in the case I am detecting the members of my family the label map looks like this:

```protobuf
item {
    id: 1
    name: 'lindo_st_angel'
}
item {
    id: 2
    name: 'nikki_st_angel'
}
item {
    id: 3
    name: 'eva_st_angel'
}
item {
    id: 4
    name: 'nico_st_angel'
}
```

Note that the id needs to start at 0. Store this file in the ```annotations``` folder with the name ```label_map.pbtxt```.

5. Create TFRecord (.record)

TFRecord is an important data format designed for Tensorflow. (Read more about it [here](https://www.tensorflow.org/tutorials/load_data/tf_records)). Before you can train your custom object detector, you must convert your data into the TFRecord format.

Since we need to train as well as validate our model, the data set will be split into training (```train.record```) and validation sets (```val.record```). The purpose of training set is straight forward - it is the set of examples the model learns from. The validation set is a set of examples used DURING TRAINING to iteratively assess model accuracy.

I used the program [create_tf_record.py](./create_tf_record.py) to convert the data set into train.record and val.record.

This program is preconfigured to do 70–30 train-val split. Execute it by running:

```bash
$ cd <path_to_your_tensorflow_installation>/models
$ python3 ./create_tf_record.py \
--data_dir ./ \
--output_dir ./tf_record
```

As configured above the program will store the ``.record`` files to the ```tf_record``` folder. 

6. Download pre-trained model

There are many pre-trained object detection models available in the model zoo but you need to limit your selection to those that can be converted to quantized TensorFlow Lite (object detection) models. (You must use [quantization-aware training](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize#quantization-aware-training), so the model must be designed with fake quantization nodes.)

In order to train them using our custom data set, the models need to be restored in Tensorflow using their checkpoints (```.ckpt``` files), which are records of previous model states.

For this example download ```ssd_mobilenet_v2_quantized_coco``` from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz) and save its model checkpoint files (```model.ckpt.meta```, ```model.ckpt.index```, ```model.ckpt.data-00000-of-00001```) to the ```models/checkpoints/``` directory.

7. Modify Config (.config) File

If required (for example you are changing the number of classes from 4 used in this example to something else) modify the files in the ```/config``` directory as needed. There should not be many changes required if using the scripts above as directed. 

8. Re-train model. 

Follow the steps below to re-train the model replacing the values for ```pipline_config_path``` and ```num_training_steps``` as needed. I found 1400 training steps to be sufficient in this example. 

```bash
$ cd <path_to_your_tensorflow_installation>/models
$ retrain_detection_model.sh \
--pipeline_config_path ./configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config \
--num_training_steps 1400
```

9. To monitor training progress, start tensorboard in a new terminal:

```bash
$ cd <path_to_your_tensorflow_installation>/models
$ tensorboard --logdir ./train/
```

10. Convert model to TF Lite and compile it for the edge tpu.

Run the following script to export the model to a frozen graph, convert it to a TF Lite model and compile it to run on the edge TPU. Replace the pipeline configuration path as required and make sure the checkpoint number matches the last training step used in training the model.

NB: this assumes the [Edge TPU Compiler](https://coral.withgoogle.com/docs/edgetpu/compiler/) has been installed on your system.

```bash
$ cd <path_to_your_tensorflow_installation>/models
$ convert_checkpoint_to_edgetpu.sh \
--pipeline_config_path ./configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config
--checkpoint_num 1400
```

11. Run the model

You can now use the retrained and compiled model with the [Edge TPU Python API](https://coral.withgoogle.com/docs/edgetpu/api-intro/).

## License

[Apache License 2.0](LICENSE)