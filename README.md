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
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```
[NOTE] - Since some of the generated output files (i.e. the checkpoints, eval and train tf events, etc.) are quite large, they won't be committed to this repo.

### Exploratory Data Analysis (EDA)

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
    rgb_mapping = { 1: 'red',   # Vehicles
                    2: 'green', # Cyclists
                    4: 'blue'   # Pedestrians
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
        for bb, cl in zip (normalized_bboxes, classes):
            y1, x1, y2, x2 = bb
            anchor_point = (x1, y1)
            bb_w =  x2 - x1
            bb_h = y2 - y1
            rec = patches.Rectangle(anchor_point, bb_w, bb_h, facecolor='none', edgecolor=rgb_mapping[cl])
            ax[x, y].add_patch(rec)
        ax[x, y].axis('off')
    plt.tight_layout()
    plt.show()

# Display 10 random images in dataset
rand_dataset = dataset.shuffle(1000)
display_images(dataset.take(10))
```

### EDA Analysis

My initial impression of the resulting images and bounding boxes from the previous section is that the current model:
  - Might have some false positives in the dataset since some images have an illegable amount of bounding boxes
  - Might struggle with smaller objects (like pedestrians and cyclists)
  - Some of the bounding boxes seem slightly off when the weather isn't perfect conditions (blurry images from rain, not perfect lighting, etc.)

To better understand the dataset, I created a get_occurence_metrics function to create a bar plot of the classification break down across the enter train folder.

```python
def get_occurence_metrics(batch_data):
    """
    This function takes the batch_data from the dataset and creates a distribution
    bar plot based on the image classifcation.
    """
    num_occurences = { 1: 0, # Vehicles
                       2: 0, # Cyclists
                       4: 0  # Pedestrians
                     }
    class_labels = [("r", "Vehicles"), ("g", "Cyclists"), ("b", "Pedestrians")]
    total_records_num_occurrences = list()
    # For each tfrecord, calculate the total number of occurences
    for i, batch in enumerate(batch_data):
        classes = batch['groundtruth_classes'].numpy()
        # Total up the number of occurences per class
        for cl in classes:
            num_occurences[cl] += 1
        # Save off the record's occurences
        total_records_num_occurrences.append(num_occurences.copy())
        # Reset the number of occurences
        for i in num_occurences:
            num_occurences[i] = 0
    
    # Prep the subplots
    _, ax = plt.subplots()
    num_classes = len(num_occurences)
    bar_width = 0.7
    # For every record, add a bar for each classification indicating the total number
    # of occurrences in an image 
    for i, occurrence in enumerate(total_records_num_occurrences):
        for j, n in enumerate(occurrence):
            ax.bar(i + j, occurrence[n], bar_width, color=class_labels[j][0], align="edge")

    # Create the plot labels
    ax.set_title("Occurences per Classification Type")
    ax.set_ylabel("Number of Occurences")
    ax.set_xlabel("Record file number")
    # Create a pretty legend
    handles = [plt.Rectangle((0 , 0), 1, 1, color=l[0]) for l in class_labels]
    ax.legend(handles, [label[1] for label in class_labels])
    plt.tight_layout()
    # Display the results in a bar plot
    plt.show()

# Draw the occurence metrics bar plot
get_occurence_metrics(dataset.take(86))
```

The resulting bar graph seemed to show that there is a largely, unequal amount of vehicle detections. With this new information, this also leads me to believe that the current model has a harder time with smaller objects.

Additionally, after reviewing the sample images for the EDA section from the Udacity website, I also saw a case where a person's face is incorrectly tagged as a cyclist (there's a pedestrian bound box around the object and looks only like a pedestrian to me). So, my initial hypothesis seem to match the additonal evalutation.

### Model Training and Evaluation

TODO: Will fill this out later

### Improving on Performances

TODO: Will fill this out later

### Creating an Animation of the Trained Model

TODO: Will fill this out later
