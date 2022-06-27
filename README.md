# Object Detection in an Urban Environment
The first project in the Self-Driving Cars Nanodegree program insisted of taking a provided pre-trained model and creating a convolutional neural network to detect and classify objects from the Waymo Open dataset.

## Importance of Object Detection for Self-Driving Cars

Precise object detection is an imperative backbone to the topic of Self-Driving Cars. Without correct detections, the autonomous vehicles wouldn't be able to make complex, navigation decisions based on on traffic laws and avoid obstacles. 

## Data

For this project, the provided Udacity workspace contained data from the [Waymo Open dataset](https://waymo.com/open/).

[NOTE] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.

## Local vs Udacity Project Workspace

At first, I attempted to setup a local instance of the project environment. However, after many hours, I abandoned that route because I couldn't figure out how to setup the prequisites (the nvidia-smi provided a lot of difficulties and refused to work) to run the Dockerfile environment.

In the end, I opted to use the Udacity Project Workspace. This also provided it's own set of complications.

## Using the Project Workspace (Jupyter Notebooks)

I ran into numerous problems running the notebooks on the Udacity workspace. One issue, that is addressed in the project notes on Udacity, is that the Firefox browser would immediately crash when using the Desktop environment.

In order to work around this, the Chromium browser was installed and launched when running the Jupyter Notebooks.

```bash
sudo apt-get update
sudo apt-get install chromium-browser
sudo chromium-browser --no-sandbox
```

After launching the chromium browser, in another terminal,

```
cd /home/workspace
./launch_jupyter.sh
```

And copied the URL from this terminal into the browser.

However, even after switching to the chromium, the notebook would constantly crash and have to be restarted. Additionally, the Desktop environment was pretty laggy. With these issues, I quickly ate up a bunch of my alloted GPU time.

## Structure

### Data

The data used for training, validation and testing from the Udacity Project Workspace is organized as follows:
```
/home/workspace/data/
    - train: contains 86 files to train the models
    - val: contains 10 files to validation the models
    - test: contains 3 files to test your model and create inference videos
```

Currently, there's outdated information within this first project. It appears that the Udacity workspaces already have the split files within train and val. The assumption is made that the files in train/ and val/ were downsampled to have one every 10 frames from 10 fps videos and the testing folder contains 10 fps video without downsampling.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - experiment0/ - Contains the pipeline for the reference run
    - experiment1/ - Contains the pipeline for adjusting the optimizer run
    - experiment2/ - Contains the pipeline for adjusting the adding more randomness in run
    - label_map.pbtxt
    ...
```
[NOTE] - Since some of the generated output files (i.e. the pretrained_model, checkpoints, eval and train tf events, etc.) are quite large, they won't be committed to this repo.

## Exploratory Data Analysis (EDA)

In order to better understand the provided data, a display_images function was implemented to help display 10 shuffled images with their correctly classified colored bounding boxes. The provided classifications and color mapping was:

| Classification | Color |
|----------------|-------|
| Vehicles       | Red   |
| Pedestrians    | Blue  |
| Cyclists       | Green |

The full implementation can also be found in Exploratory Data Analysis.ipynb:
```python
from utils import get_dataset
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

%matplotlib inline

dataset_glob_path = "/home/workspace/data/train/*.tfrecord"
dataset = get_dataset(dataset_glob_path)

def display_images(batch):
    """
    This function takes a batch from the dataset and displays the image with
    the associated bounding boxes.
    """
    rgb_mapping = {
        1: 'red',   # Vehicles
        2: 'blue',  # Pedestrians
        4: 'green'  # Cyclists
    }

    col, row = 5, 2
    _, ax = plt.subplots(col, row, figsize=(36, 36))
    for n, batch_data in enumerate(batch):
        x = n % col
        y = n % row
        # Parse out display information
        bboxes = batch_data['groundtruth_boxes'].numpy()
        classes = batch_data['groundtruth_classes'].numpy()
        img = batch_data['image']
        img_height, img_width, _ = img.shape

        # Display the batch image
        ax[x, y].imshow(img)

        # Normalize the bounding boxes to the current image size
        normalized_bboxes = copy.deepcopy(bboxes)
        normalized_bboxes[:, (0, 2)] = bboxes[:, (0, 2)] * img_height
        normalized_bboxes[:, (1, 3)] = bboxes[:, (1, 3)] * img_width

        # Draw the bounding box with the correct coloring based on the class
        for bb, cl in zip(normalized_bboxes, classes):
            y1, x1, y2, x2 = bb
            anchor_point = (x1, y1)
            bb_w = x2 - x1
            bb_h = y2 - y1
            rec = patches.Rectangle(anchor_point, bb_w, bb_h, facecolor='none',
                                    edgecolor=rgb_mapping[cl])
            ax[x, y].add_patch(rec)
        ax[x, y].axis('off')
    plt.tight_layout()
    plt.show()

# Display 10 random images in dataset
rand_dataset = dataset.shuffle(1000)
display_images(dataset.take(10))
```

![10 Images with bounding boxes](images/ten_bb_images.PNG "Ten Shuffled Images with Bounding Boxes")

### EDA Analysis

My initial impression of the resulting images and bounding boxes from the previous section is that the current model:
  - Might have some false positives in the dataset since some images have an illegable amount of bounding boxes
  - Might struggle with smaller objects (like pedestrians and cyclists)
  - Some of the bounding boxes seem slightly off when the weather isn't perfect conditions (blurry images from rain, not perfect lighting, etc.)

To better understand the dataset, I created a get_occurrence_metrics function to create a pie chart of the classification break down across the entire train folder.

```python
def get_occurrence_metrics(batch_data):
    """
    Creates a pie chart based on the class distribution
    of the provided batch data
    """
    num_occur = {
        1: 0,  # Vehicles
        2: 0,  # Pedestrians
        4: 0   # Cylists
    }
    class_labels = ["Vehicles", "Pedestrians", "Cyclists"]
    colors_ref = ['r', 'b', 'g']
    # For each tfrecord, calculate the total number of occurrences
    for batch in batch_data:
        classes = batch['groundtruth_classes'].numpy()
        unique, counts = np.unique(classes, return_counts=True)

        for u, c in zip(unique, counts):
            num_occur[u] += c

    # Create a pie chart of the class occurrences
    _, ax = plt.subplots()
    ax.pie(list(num_occur.values()), labels=class_labels, colors=colors_ref,
           autopct='%1.1f%%', shadow=True)

    plt.show()


# Draw the occurrence metrics pie chart
get_occurrence_metrics(dataset.take(86))
```

![Occurrences Pie Chart](images/occurrences_pie_graph.PNG "Class occurrences")

The resulting pie chart shows there is a largely, unequal amount of vehicle detections. With this new information, this also leads me to believe that the current model has a harder time with smaller objects and potentially over generalizing.

Additionally, after reviewing the sample images for the EDA section from the Udacity website, I also saw a case where a person's face is incorrectly tagged as a cyclist (there's a pedestrian bound box around the object and looks only like a pedestrian to me). So, my initial hypothesis seem to match the additonal evalutation.

## Model Training and Evaluation

### Tensorflow Pipeline Config

The provided reference pipeline config used a Single Shot Detector (SSD) Resnet 50 640x640 model. The SSD paper can also be found [here](https://arxiv.org/pdf/1512.02325.pdf).

### Experiment 0 (Reference run)

The first portion of this section contained instructions on a reference experimentation. This was done by creating a training process:

```bash
python experiments/model_main_tf2.py --model_dir=experiments/reference --pipeline_config_path=experiments/reference/pipeline_new.config
```

To visualize the training, I launched a tensorboard instance by running:

```bash
python -m tensorboard.main --logdir experiments/reference
```

After the training was finished the following images were captured:

![Experiment 0 Reference Loss Image](images/experiment0_ref_training_model.PNG "Experiment 0 Reference Loss")

![Experiment 0 Learning rate and steps per second Image](images/exp0_ref_training_model_2.PNG "Experiment 0 Learning rate and steps per second")

As seen from the images above, the loss function indicates, from all the non-smooth lines and the early plateauing learning rate, that the model could firstly benefit from learning annealing.

Following this, an evaluation process was launched:

```bash
python experiments/model_main_tf2.py --model_dir=experiments/reference --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference
```

This wasn't described in the project instructions but after realizing a single blue dot point for the Detection boxes seemed off, I found a similar question on the Udacity Knowledge forums where after each time I hit, "Waiting for next checkpoint", I edited the experiment/reference/checkpoint file's 'model_checkpoint_path' field to increment the ckpt-<x> option (i.e. ckpt-1 through ckpt-6).

This generated the following evaluations:

![Reference Detection Boxes Precision Image](images/reference_detection_boxes_precision.PNG "Reference Detection Boxes Precision")

![Reference Detection Boxes Recall Image](images/reference_detectionboxes_recall.PNG "Reference Detection Boxes Recall")

I realized after doing Experiment 1 that you also need to edit the checkpoint file first to do ckpt-1 and increment sequentially or the charts come out wrong.

![Reference Step Time and Loss](images/ref_step_time_and_loss.PNG "Reference Step Time and Loss Metrics")

The step time and loss from the terminal output also indicate that some improvements need to be made to the learning rate and reducing the loss.

## Improving on Performances

### Experiment 1 (Adjusting the Optimizer)

As found from the first run, the model's pipeline could benefit from learning annealing. After some research, I learned that using the Adam optimizer might help improve the learning rate. Additonally the learning rate seems a little high. So, I reduced it by a factor of 1e-1.

The changes are as follows:

```
optimizer {
    adam_optimizer: {
      epsilon: 1e-7
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 0.004
          total_steps: 2500
          warmup_learning_rate: 0.0013333
          warmup_steps: 200
        }
      }
    }
    use_moving_average: false
}
```

By adjusting this, it made a large difference in the loss and learning rate:

![Experiment 1 Loss Image](images/exp1_optimizer_loss_train.PNG "Experiment 1 Loss")

![Experiment 1 Learning rate and steps per second Image](images/exp1_optimizer_learning_rate_and_sps.PNG "Experiment 1 Learning rate and steps per second")

![Experiment 1 Step Time and Loss](images/exp1_optimizer_step_time_and_loss.PNG "Experiment 1 Step Time and Loss Metrics")

I lost the reference detection precision and recall evaluations in between this experiment and the next. However, the results were better than the reference as well.

![Experiment 1 Detection Boxes Precision Image](images/exp1_optimizer_detection_precision.PNG "Reference Detection Boxes Precision")

![Experiment 1 Detection Boxes Recall Image](images/exp1_optimizer_detection_recall.PNG "Reference Detection Boxes Recall")

### Experiment 2 (Adding Randomness)

The SSD paper also indicates that, like many other Machine Learning algorithms, it can have a hard time detecting smaller objects (pg 8, 12). Potentially, this could be the reason why there's a disproportionate amount of vehicle detections to small objects like cyclists and pedestrians. However, our dataset might just be lacking images that have many of these objects.

As suggested in the instructions, I attempted to adjust the data augmentation, albumentation, strategy to attempt to help diversify the training set to help with overgeneralization by the model. By default, Tensorflow provides a few different strategies which can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto). In the SSD paper, the authors found that random cropping and changing the fixed aspect ratio greatly improves the results (pg 4-6, 10). The original pipeline already had some random cropping and flipping --- which I left.

I also noticed from the EDA section that the model seemed to be performing worse when conditions weren't as optimial. So, for this experiment, I also added in random adjusting brightness and saturation.

Therefore, I adjusted the pipeline to:

```
  data_augmentation_options {
    random_adjust_brightness {
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
    }
  }
  data_augmentation_options {
    random_square_crop_by_scale {
      scale_min: 0.7
      scale_max: 1.4
    }
  }
```

Unfortunately, my hypothesis for this experiment didn't match the results I was expecting:

![Experiment 2 Loss Image](images/exp2_randomness_loss.PNG "Experiment 2 Loss")

![Experiment 2 Step Time and Loss](images/exp2_randomness_step_time_and_loss.PNG "Experiment 2 Step Time and Loss Metrics")

Also, upon reaching step 2500, my workspace ran out of room. So, the six checkpoint was corrupted. However, even without it, the loss function was much worse indicating that I over randomized the pipeline. It does still outperform the reference experiment though.

Additionally, The learning rate and the improving loss towards the end did indicate that this step might also benefit from additional stepping. Interestingly enough, the steps per second seemed to be marginally better. With more time, I would probably re-test the first experiment to see if that metric stayed where it was at.

![Experiment 2 Learning rate and steps per second Image](images/exp2_randomness_learning_rate_and_sps.PNG "Experiment 2 Learning rate and steps per second")

The detection precision and recall graphs also indicate that this setup didn't improve on Experiment 1.

![Experiment 2 Detection Boxes Precision Image](images/exp2_randomness_detection_precesion.PNG "Experiment 2 Detection Boxes Precision")

![Experiment 2 Detection Boxes Recall Image](images/exp2_randomness_detection_recall.PNG "Experiment 2 Detection Boxes Recall")

## Summary

Although the second experiment didn't result in better results, the data seem to indicate that both experiment one and two out perform the reference. 

![Summary learning rate and steps per second](images/summary_learning_rate_sps.PNG "Summary learning rate and steps per second")

![Summary loss](images/exp2_randomness_detection_recall.PNG "Summary loss")

With more time, I would try to fine tune the first experiment some more by reducing some of the randomness that I added in from experiment 2. Additionally, I would attempt to experiment with a different algorithm like Faster-RCNN to see how it pairs up to the SSD with this dataset.
