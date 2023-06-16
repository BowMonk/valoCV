# Comparative Evaluation of YOLO and SAM+GroundingDINO for Aimbot Development in Valorant: A Computer Vision Perspective

**Members:**
Arbër Demi 5073227
Akash Amalan 4682505

### Introduction

Aim bots, an intersection of artificial intelligence (AI) and gaming, represent an intriguing area of study. Their study unveils innovative algorithms, novel training methodologies, and enhanced performance in intricate gaming environments. On a different note, the use of aim bots often leads to unfair gaming practices, which prompts the necessity of exploring this domain to ensure ethical gaming experiences.

Existing solutions for automated gameplay primarily rely on metadata acquired directly from the game. However, this approach lacks creativity and diversity in operations, as it mainly revolves around the basic change of mouse positioning. Enter the world of computer vision - with AI models like YOLO ("You Only Look Once") making strides in object detection. But can we do better? To answer this question, we aim to develop a model combining the capabilities of SAM (Segment Anything Model) and GroundingDINO, and compare its performance with the renowned YOLO model.

We chose Valorant, a popular first-person shooter game by Riot Games, as our testbed due to its dynamic environment and vast user base. Our model targets enemy heads and bodies to evaluate its performance.

### Methodology


Our prime objective is to examine how the combined SAM+GroundingDINO model stacks up against the traditional YOLO model. SAM, an object segmentation model developed by Meta, coupled with GroundingDINO, a zero-shot object detection model, is a promising combination. We hypothesize that this could provide improved accuracy and robust behavior masking for vision-based bots.


For our experiment, we leveraged pre-annotated labels and training/test images from publicly available datasets on Roboflow. These datasets include:

- [Valorant Object Detection](https://universe.roboflow.com/valorantobjectdetection/valo-1h0lc)
- [Project Valorant](https://universe.roboflow.com/project-nqyj3/valorant-f3198)
- [Santyasa Dataset](https://universe.roboflow.com/alfin-scifo/santyasa/dataset)

In the following sections, we will detail how these datasets were utilized for training and validating our models.


### Resources
The experiments were done on two separate CPUs and GPUs. 

| Experiment Model       | CPU                      | GPU |
|------------------------|--------------------------| --- |
| **YOLO**                   | Intel Core I5 12-600k    | NVIDIA RTX 3070 |
| **Grounding Dina and SEM** | Intel Core I9 11-11900KF | NVIDIA RTX 3070TI |



### Datasets
Each dataset comes with a training set, validation set, and testing set. The images are provided in JPEG format, while the labels for each set are stored in separate text files. 

The label format is as follows:

`2 0.234375 0.47716346153846156 0.03125 0.040865384615384616`

`1 0.2644230769230769 0.5685096153846154 0.08173076923076923 0.11298076923076923`

Here, the first integer indicates the class, and the rest of the numbers represent the coordinates of the bounding box. Each label set thus provides crucial information for the models to correctly identify and locate the objects within the images.
the first integer represents the class and the rest of the coordinates of the bounding box. 

Do note that all these datasets contain different labels with different quality of images. Here are some random examples of all 3 datasets.

<h2 align="center">Dataset 1</h2>
<div align="center">
<table>
  <tr>
    <td align="center"><img src="images/dataset1/11.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></td>
    <td align="center"><img src="images/dataset1/12.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></td>
    <td align="center"><img src="images/dataset1/13.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset1/14.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></td>
    <td align="center"><img src="images/dataset1/15.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></td>
    <td align="center"><img src="images/dataset1/16.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset1/17.jpg" width="200px;" alt="Image7"/><br /><sub><b>Image 7</b></sub></td>
    <td align="center"><img src="images/dataset1/18.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></td>
    <td align="center"><img src="images/dataset1/19.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></td>
  </tr>

</table>
</div>


<br>
<br>

<h2 align="center">Dataset 2</h2>


<div align="center">
<table>
  <tr>
    <td align="center"><img src="images/dataset2/21.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></td>
    <td align="center"><img src="images/dataset2/22.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></td>
    <td align="center"><img src="images/dataset2/23.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset2/24.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></td>
    <td align="center"><img src="images/dataset2/25.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></td>
    <td align="center"><img src="images/dataset2/26.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset2/27.jpg" width="200px" alt="Image7"/><br /><sub><b>Image 7</b></sub></td>
    <td align="center"><img src="images/dataset2/28.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></td>
    <td align="center"><img src="images/dataset2/29.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></td>
  </tr>

</table>
</div>

<br>
<br>

<h2 align="center">Dataset 3</h2>



<div align="center">
<table>
  <tr>
    <td align="center"><img src="images/dataset3/31.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></td>
    <td align="center"><img src="images/dataset3/32.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></td>
    <td align="center"><img src="images/dataset3/33.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset3/34.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></td>
    <td align="center"><img src="images/dataset3/35.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></td>
    <td align="center"><img src="images/dataset3/36.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset3/37.jpg" width="200px;" alt="Image7"/><br /><sub><b>Image 7</b></sub></td>
    <td align="center"><img src="images/dataset3/38.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></td>
    <td align="center"><img src="images/dataset3/39.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></td>
  </tr>

</table>
</div>

<h2 align="center">Dataset 4</h2>

Our final dataset is a video that showcases the performance of real-time detection in action. This video features a streamer (TenZ) practicing in the in-game shooting range. In this scenario, enemies spawn in rapid succession and must be neutralized before they disappear.

<p align="center">
<img src='pics/video_example.jpg' width='600'>
</p>
To facilitate our evaluation, we annotated all frames in which an enemy appears with dots centering on the enemy's head and body, as illustrated below:

<p align="center">
<img src='pics/shooting annotation.png' width='600'>
</p>
During detection moments, we took the initiative to reduce on-screen clutter by whiting out the camera and the ad banner in the images.
The use of these points will be detailed further in the online evaluation section.


#### Comparison

It is apparent that dataset 1 has high quality images while dataset 2 and 3 has low quality in-game images. One can decide to remove some of the images
of dataset 1 and 2 due to its resolution. However, we approach this differently for both Yolo and GroundingDino with SIEM as we will explain in the next sections.

## Yolo

YOLO, or "You Only Look Once," has redefined the realm of computer vision with its high-accuracy, real-time object detection system. Unlike traditional object detection systems that propose multiple regions and classify them independently, YOLO converts this into a single regression problem. It breaks down an image into a grid, with each cell responsible for predicting bounding boxes and class probabilities, thereby considerably enhancing processing speed.

The YOLO architecture utilizes a Convolutional Neural Network (CNN) that takes an entire image as input and predicts bounding boxes and class probabilities directly in a single pass. The backbone of this network comprises 24 convolutional layers and two fully connected layers. It outputs a tensor with bounding box properties and class probabilities, where a final score for each box is computed, and boxes with scores below a certain threshold are discarded.

YOLO's architecture allows it to process the whole image during training, enabling it to encode contextual information about classes and their appearance. This enhances its ability to detect objects of various sizes and in different settings. Over time, YOLO's architecture has seen improvements like skip connections, upsampling, and the use of anchor boxes to refine bounding box predictions, ensuring it remains both fast and accurate.

### YOLO Versions and Variants: Choosing the Right One for Our Project

Navigating through the diverse landscape of object detection algorithms, one cannot miss the dynamism and robustness of YOLO (You Only Look Once)[1]. Known for its speedy real-time inference, YOLO has seen multiple iterations, each carving a niche for itself with unique capabilities and optimizations.

The spotlight often falls on YOLOv3, a common choice among researchers and developers alike. However, the YOLO universe doesn't stop there. The emergence of newer versions like YOLOv7 and YOLOv8 underscores continuous innovations in the pursuit of faster and more efficient real-time object detection[1].

For this project, our journey led us to YOLOv5, an intriguing balance between performance and computational resource consumption. But the YOLOv5 story isn't one-size-fits-all, with different variants ('s', 'm', 'l', and 'x') offering a spectrum of speed and accuracy trade-offs[2].

- The 's' variant, being the smallest and fastest, tempts with high speed but at the expense of accuracy.
- The 'm' variant, our chosen companion for this endeavor, strikes a harmonious balance between speed and accuracy.
- The 'l' and 'x' variants, on the other end of the spectrum, offer higher accuracy but at the cost of slower inference speed.

The choice of the 'm' model wasn't random[2]. It emerged as the optimal solution considering:

1. **Computational Resources**: The 'm' variant adeptly balances computational efficiency and accuracy. While the 'l' and 'x' could have offered higher accuracy, their larger computational appetite wouldn't fit the bill for our use-case.
2. **Real-Time Performance**: With our application deeply rooted in the fast-paced dynamics of Valorant gameplay, maintaining real-time performance was paramount. The 'm' variant checked this box effectively.
3. **Dataset Complexity**: Our datasets brought with them their own set of challenges, with varying complexities and quality. The 'm' model showcased its robustness in handling diverse image resolutions and noise levels.

Remember, the choice of the model variant significantly hinges on your specific use case and available resources. If your priority is high accuracy and computational resources aren't a constraint, the larger 'l' or 'x' models might just be your ideal match. However, for this project, 'm' proved to be our optimal partner.
This is the overall architecture of Yolo5m:

<p align="center">
  <img src="images/yolo.png" alt="Yolo Im">
</p>





#### Training and Validation: Harmonizing Datasets and Embracing Diversity

For training YOLO, we utilized all three distinct datasets, each with its unique characteristics but adhering to a consistent format. The quality of images within these datasets varied, as did the labels they utilized.

All datasets encompassed `EnemyHead` and `EnemyBody` labels, which were our primary focus for this study. However, they also contained additional labels such as `BoomBot` and other game-specific abilities. These extra labels, although interesting, did not align directly with our current objective.

To achieve a streamlined training process, we undertook the task of re-labeling these datasets. 
This harmonization allowed us to align the data more closely with our goal.
We adopted the following label convention: `['boomybot', 'enemyBody', 'enemyHead', 'smoke', 'splat', 'teammate', 'walledoff enemy']`. 
This decision was guided by the inner workings of YOLO. 

YOLO operates based on class predictions rather than class names, translating our meticulously chosen labels into a plain numerical list: `[0, 1, 2, 3, 4, 5, 6, 7]`. Through our re-labeling effort, we ensured that every class number correctly corresponded to its respective label across all datasets.

One might question our decision to use a wide range of image qualities, rather than solely high-resolution images or curating a pruned selection. The answer lies in the trade-off between quality and diversity. Incorporating images of varying quality level challenged YOLO's ability to interpret images under different conditions. 
This not only enriched our training data but also aimed to enhance YOLO's generalization performance on the test set,
crafting a more robust model ready to tackle different angles one might encounter in Valorant. Therefore all 3 datasets was merged into one
and used for training the Yolo model.

##### Process and Commands
The training process is initiated through a command-line interface, which accepts a set of parameters: the image size, batch size, number of epochs, a data.yaml file specifying the dataset configuration, the chosen model, and the weights.

The data.yaml file is a fundamental piece that provides details about the data being used. An example of this file might look like this:

```yaml
path: datasets/dataset2
train: train/images
val: valid/images
test: test/images
nc: 7
names: ['boomybot', 'enemyBody', 'enemyHead', 'smoke', 'splat', 'teammate', 'walledoff enemy']
```

In this YAML file, we specify the path of our dataset, the locations of the training, validation, and testing images, the number of classes (`nc`), and the names of these classes.

The validation process follows a similar structure and is run using the following command:

```shell
python val.py --weights ../best.pt --data valorant.yaml --img 416
```

Finally, the testing process is also similar but requires an additional source parameter indicating the path to the testing dataset, and a confidence threshold for the bounding boxes. It can be executed using the following command:

```shell
python detect.py --weights ../best.pt --img 640 --conf 0.25 --source datasets/dataset1/test/images
```
This structure allows for a great deal of flexibility and customization in the training, validation, and testing processes, accommodating various types of data configurations and use cases.

For validation a similar command is used:
```
python val.py --weights ../best.pt --data valorant.yaml --img 416
```

For testing also a similar command is used
```
python detect.py --weights ../best.pt --img 640 --conf 0.25 --source datasets/dataset1/test/images
```
Except you need a provide an extra source with the path to the test dataset and a confidence for the bounding boxes. 



#### Hyperparameter Tuning:


To fine-tune the model, we used an evolutionary approach. The fitness of the model, which we aim to maximize, is defined by a weighted combination of metrics in YOLOv5. Specifically, mAP@0.5 contributes 10% of the weight, and mAP@0.5:0.95 provides the remaining 90%. Precision (P) and Recall (R) are not included.

The model is trained for 100 epochs with the evolutionary approach:

```bash
python train.py --epochs 100 --data dataset --weights yolov5m.pt --cache --evolve
```

Genetic operators such as crossover and mutation are primarily used during this evolution.

The final hyperparameters are as follows:

```yaml
lr0: 0.01                  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2                   # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937            # SGD momentum/Adam beta1
weight_decay: 0.0005       # optimizer weight decay 5e-4
warmup_epochs: 3.0         # warmup epochs (fractions ok)
warmup_momentum: 0.8       # warmup initial momentum
warmup_bias_lr: 0.1        # warmup initial bias lr
box: 0.05                  # box loss gain
cls: 0.5                   # cls loss gain
cls_pw: 1.0                # cls BCELoss positive_weight
obj: 1.0                   # obj loss gain (scale with pixels)
obj_pw: 1.0                # obj BCELoss positive_weight
iou_t: 0.20                # IoU training threshold
anchor_t: 4.0              # anchor-multiple threshold
fl_gamma: 0.0              # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015               # image HSV-Hue augmentation (fraction)
hsv_s: 0.7                 # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4                 # image HSV-Value augmentation (fraction)
degrees: 0.0               # image rotation (+/- deg)
translate: 0.1             # image translation (+/- fraction)
scale: 0.5                 # image scale (+/- gain)
shear: 0.0                 # image shear (+/- deg)
perspective: 0.0           # image perspective (+/- fraction), range 0-0.001
flipud: 0.0                # image flip up-down (probability)
fliplr: 0.5                # image flip left-right (probability)
mosaic: 1.0                # image mosaic (probability)
mixup: 0.0                 # image mixup (probability)
copy_paste: 0.0            # segment copy-paste (probability)
```

These hyperparameters determine the behavior of the training process and significantly 
influence the model's performance. Fine-tuning these parameters is an essential part of computer vision
pipeline to ensure optimal performance of the model on the task at hand.

---

### GroundingDINO and SAM [2][3]

<p align="center">
  <img src="pics/dino_arch.png" alt="Yolo Im">
</p>

#### Description
GroundingDINO is a novel zero-shot object detection model that has great performance in existing datasets.
It is a combination of many different sections coming together to make a great framework mainly used for annotation
and segmentation, in conjunction with SAM.

SAM(Segment Anything Model) is a recent AI model developed by Meta. It focuses on image segmentation
and in conjunction with GroundingDINO, it makes for a nice pipeline to detect and create masks for objects 
in view.

For more information on this specific use case of GroundingDINO and SAM, this Medium article does a great job at giving 
more information: https://pub.towardsai.net/grounding-dino-achieving-sota-zero-shot-learning-object-detection-78388d6842ed.
It is also the source of the architecture image shown above.
#### Annotations
The annotations of the datasets we used are not very compatible with GroundingDINO and SAM. The annotations are small boxes which fit well with the detection style of YOLO and are seemingly annotated with that in mind.
However, we had trouble getting GroundingDINO to get a similar detection style, simply because it does not use the same grid system that YOLO uses, and the textual prompts trying to isolate some areas do not work well all the time.

The annotations were specifically made for two labels: "Head" and "Person".

To make sure that the datasets were useful for both sides, we manually annotated the datasets, except for the third one as it was too noisy to make consistent decisions on annotating.
For this annotation procedure we used labelme (the package can be installed through pip), and manually annotated boxes that would fit GroundingDINOs detection style.
Some examples of this annotation can be seen below:

<p align="center">
<img src='pics/annotation_example_1.png' height='200'>
<img src='pics/annotation_example_2.png' height='200'>
<p>

<p align="center">
<img src='pics/annotation_example_3.png' height='200'>
<img src='pics/annotation_example_4.png' height='200'>
<p>

To make the annotations usable for training and evaluation, they had to be converted to a format that we were already using.
The steps shown at https://github.com/Tony607/labelme2coco were followed to turn the annotations into COCO data formatted JSON.
#### Training

As GroundingDINO is fairly new (March 2023), the training pipeline for it is not released yet. With the resources
available online, at the moment we are only able to change our labels and tune some thresholds.

When it comes to labels, DINO can handle complex label prompts such as "man sitting on a chair", however it has a lot 
of false positives when it comes to objects that it is not properly tuned for. This was made apparent quite quickly when 
we started to mess around with labels as results would change even depending on what other labels were included, for 
example, some false positives for the label "head" would disappear if the model was also prompted to detect "shoe". This
had something to do with the confidence thresholds in-place, however the exact reason for these changes is still a bit 
unclear.

The biggest influence when it comes to detection was coming from the text and box confidence threshold parameters.
These are passed to the model before it does detections to decide what is good enough as a detection. How we decided
on what is a best threshold was with a simple search with MAP evaluation on the first dataset (the best performing thresholds were chosen).
This search was done using different combinations of box and text thresholds, with a range from 0.05 to 0.45 (thresholds higher than 0.45 were eliminating too many clear cases)
In the end, the best performing cases were around 0.4 to 0.45, however there were many more cases with no detections.
So instead we settled on 0.35 for both with an MAP of 0.803, as it detected nearly all cases.


### Offline Evaluation

The key to any comparative study lies in its evaluation metrics. We use Mean Average Precision (MAP), Precision, Recall, and F1 score as our key metrics to evaluate the performance of the models 
only on the images from the datasets(not videos):

- **Precision** is the ratio of correctly predicted positive observations to the total predicted positives.
- **Recall** (Sensitivity) calculates the ratio of correctly predicted positive observations to the all observations in actual class.
- **F1 Score** is the weighted average of Precision and Recall. It tries to find the balance between precision and recall.
- **MAP** (Mean Average Precision) is used in information retrieval to measure the effectiveness of the model in terms of precision of retrieval over a range of recall values.

Do note that **accuracy** was not considered due to the several reasons:
 -  The number of pixels representing the object (foreground) can be much smaller than the pixels representing the background. In such cases, a model that only predicts the background would still have a high accuracy, despite failing to detect any objects. 
 -  Accuracy doesn't account for localization errors. A model may correctly identify the presence of an object but inaccurately define its boundaries or location. Accuracy would still consider this a correct prediction, even though the location is wrong.
 -  Multiple Objects: In scenarios where multiple instances of an object can exist in a single image, accuracy as a metric falls short. If a model fails to detect one of several objects, the accuracy might still be high despite the missed detection.
 -  Precision and Recall Tradeoff: Accuracy doesn't account for the tradeoff between precision (how many selected items are relevant) and recall (how many relevant items are selected). In some contexts, it may be more important to prioritize one over the other.

### Online Evaluation
In the fast-paced world of online gaming, real-time performance is critical. We're testing our system on two key parameters – accuracy in head classification and inference speed.
For accuracy, we're using annotated videos from Dataset 4, comparing our models detection with ground truth  i.e check if the center point of the ground truth lie in the predicted bounding box. It's a real-time check on whether our system accurately recognizes heads in live gaming.
For speed, we're timing how quickly our model predicts based on each frame. It is important to note that
eventhough we use a video ,  we load it as a stream of bytes frame by frame to replicate live feed.
We also account for any target movement during this prediction time. So, we're essentially striving to ensure the models can accurately and swiftly detect dynamic movements in live game feeds.



### Expected Outcomes

Through this experiment, we aim to answer the following questions:

- How does YOLO compare to SAM+GroundingDINO for  metrics: MAP, recall, precision, F1 score?
- What hyperparameters need to be tuned between the two models?
- How well does the model react to dynamic movements in the online game feed?
- Can SAM+GroundingDINO outperform YOLO in a dynamic first-person shooter game like Valorant?

Our study aims to shed light on the applications of advanced computer vision techniques in gaming, specifically first-person shooters. By comparing different object detection architectures, we hope to contribute to the ongoing development of fair and ethical gaming practices.

### Results

#### Yolo
The results for Yolo for both offline performance and online performance are presented below

##### Offline performance

###### Sample Predictions 

Before diving into diving metrics, it is nice to look at  sample predictions in a batch from yolo.

<p align="center">
    <b>Ground Truth</b><br>
    <img src="images/badpics/val_batch2_labels.jpg">
</p>

<p align="center">
    <b>Predictions</b><br>
    <img src="images/badpics/val_batch2_pred.jpg">
</p>

At first glance, it seems the Yolo was successfully able to predict perfectly from this batch. However, it is not 
100% perfect, we have some interesting cases from different batches that are presented below:

<p align="center">
<b>Bad Fruits</b>

| ![Image 1](images/badpics/img.png) | ![Image 2](images/badpics/img_1.png) | ![Image 3](images/badpics/img_2.png) |
|:---:|:---:|:---:|
| ![Image 4](images/badpics/img_3.png) | ![Image 5](images/badpics/img_4.png) | ![Image 6](images/badpics/img_5.png) |
</p>



As we delve into the intricacies of yolo's behavior, we uncover fascinating quirks in its operation. Take, for instance, occasions when the model tags a firearm as the body of an adversary or even overlooks an enemy situated directly in front of the player. These scenarios provide a tantalizing glimpse into the thin line our model treads between correct and incorrect identification.

###### Metrics 
_Table I_

| Class       | Images | Instances |    P    |    R    | mAP50  | mAP50-95 | F1-score(threshold = 0.5) |
|-------------|--------|-----------|---------|---------|--------|----------|---------------------------|
| all         |  243   |   541     |  0.924  |  0.872  | 0.911  |  0.57    | 0.79                      |
| enemyBody   |  243   |   252     |  0.978  |  0.968  | 0.984  |  0.702   | 0.94                      |
| enemyHead   |  243   |   289     |  0.87   |  0.775  | 0.838  |  0.438   | 0.83                      |

In this table, "Class" refers to the object being detected, "Images" is the number of images used, 
"Instances" is the number of instances the class appeared in the dataset, "P" is precision, "R" is recall, 
"mAP50" is mean average precision at 50% Intersection over Union (IoU) and
"mAP50-95" is mean average precision at IoU from 50% to 95%.


It is apparent from the high precision and recall for enemyBody shows that it can in most realisitc cases 
be able to draw the correct bounding boxes. It is also apparent that it is much easier to detect the enemyBody than the 
head which is something that we would expect. Nonetheless, this model performs very well in detecting enemyHead, in more than 75%
of the cases the head is correctly recognized in the validation set as the metrics mAP and F1-score reveal at a threshold
or confidence of 0.5.


_Figure 1: Metrics_

| ![R_curve](images/metrics/R_curve.png) | ![PR_curve](images/metrics/PR_curve.png) |
|:--------------------------------------:|:----------------------------------------:|
| ![P_curve](images/metrics/P_curve.png) | ![F1_curve](images/metrics/F1_curve.png) |

Presented in the Figure 1 are four intriguing charts - Confidence vs Recall, Confidence vs Precision/Recall, Confidence vs Precision, and Confidence vs F1-score. These visualizations offer a glimpse into our model's confidence in relation to various metrics. It's this information that helped us optimize the model's confidence threshold.

When we examine the Precision/Recall graphs, it appears that a confidence level of 0.7 or more would be acceptable. Yet, the F1-Confidence graph suggests an optimal confidence threshold around 0.5. It was this seemingly counter-intuitive revelation that guided us in setting our model's inference time confidence level.
But that's not where our explorations ended! We were also intrigued to observe how the performance metrics evolved as the model processed increasing numbers of images. To this end, we visualized the changes over a batch of 700 images.

_Figure 2: Training Loss and Evolution_

![results.png](images/metrics/results.png)
The resulting charts corroborate the idea that our model is on the right learning track - 
the training loss continually diminishes as the model digests more images. 
Simultaneously, precision and recall experience fluctuations but maintain an upward trajectory on average. Similarly, the mAP_0.5 and mAP_0.5:0.95 values also demonstrate a steady rise, further affirming our model's expected learning curve. It's refreshing to observe that our model is progressing just as anticipated, devoid of any unwelcome surprises.

##### Online performance
Yolo performed pretty well in real time detection. The inference delay was measured by seeing how fast the model is able to predict before the object to the next frame.
The average delay of detecting object per frame was 0 with the maximum delay being 50 ms. This meant that the Yolo model was able to keep up 30 fps live feed. Furthermore
accuracy was also calculated by checking if the ground truth was inside the predicted bounding box. The accuracy was found to be 1 meaning that Yolo 
is not only fast but also very accurate. Similarly, recall and precision was also 1 since it was able to completely predict the ground truth.


The following gif shows you the performance of Yolo on the video of dataset 4 loaded as a stream:
<p align="center">
  <img src="live_yolo.gif" alt="Online Performance">
  <br>
  <em>Live Inference</em>
</p>

Note that the green bounding boxes are from the predicted labels, while the blue circles are from the ground truth.
If you wish to see the inference on the original video, run `main.py` under the 'yolo' folder.

The gif shows the power of Yolo and how fast the inference of each frame is. Do note that this 
was done on Yolo v5, higher versions may produce much faster results. 

#### GroundingDINO and SAM

The results for GroundingDINO and SAM are with respect to their offline performance on dataset 1 + 2, and also the 
online performance on dataset 4.

##### Offline performance

###### Sample Predictions 


In the pictures below you can see some examples of the annotations (on the left) and predictions of DINO + masking of SAM(on the right).

| ![Image 1](pics/result_anno_1.jpg) | ![Image 2](pics/result_pred_1.jpg) |
|:----------------------------------:|:----------------------------------:|
| ![Image 3](pics/result_anno_2.jpg) | ![Image 4](pics/result_pred_2.jpg) |

Although it may seem not too bad at first glance, there are quite some odd cases with DINO.

| ![Image 1](pics/dino_mess_1.jpg) | ![Image 1](pics/dino_mess_2.jpg) |
|:--------------------------------:|:--------------------------------:|
| ![Image 1](pics/dino_mess_3.jpg) | ![Image 1](pics/dino_mess_4.jpg) | 

As can be seen on the pictures, sometimes the head (marked red in the first picture) is the same as the entire body, might 
be part of the map, can include the entire torso, or even be part of the gun that the player is holding.

###### Metrics 

Table II

|    Class    |   MAP   |   AP    | Recall  | Precision | F1_Score |
|:-----------:|:-------:|:-------:|:-------:|:---------:|:--------:|
|    Head     | 0.62811 | 0.32946 | 0.27487 |  0.67112  | 0.39001  |
| Person/Body | 0.62811 | 0.92677 | 0.49863 |  0.97919  | 0.66077  |

At first glance, the most noticeable thing is the disparity between the AP (Average Precision) between the Head and Person/Body classes.
DINO was much better at detecting the general body, which is understandable as the head has less detail and is generally
harder to detect due to the smaller amount of pixels.
The model in general has low recall but higher precision, while overall not a great performance, much less than we expected.


##### Online performance

When it comes to online performance, our main measure is delay during the real-time detection.

Sadly, GroundingDINO and SAM performed quite poorly.
The average delay of detecting object per frame was around 1.11202 seconds. That is a very long time, even for a 30 FPS video.
In a real gaming environment, FPS in shooter games is quite important, and having a bot that can shoot every 65 frames or so
if the gaming is running at 60 FPS, is quite bad. This performance was slightly improved when SAM was taken out of the picture,
however DINO on its own was still slow.

Despite this, when it comes to accuracy it performed similarly to the offline setting in general accuracy, having difficulty 
with the Head class and consistently detecting the Person/Body class with a slight boost, due to less distance from the 
camera to the enemies in the shooting range.

The exact values for MAP, AP, Recall, Precision and F1_Score were not calculated due to the video not having any annotations
for the boxes (the rest of the annotation took a considerable amount of time).

The following gif shows you the performance of Dino on the video of dataset 4 loaded as a stream:
<p align="center">
  <img src="dino_live.gif" alt="Online Performance">
  <br>
  <em>Live Inference</em>
</p>

Note that the green bounding boxes are from the predicted labels, while the blue circles are from the ground truth.
As you can see from the gif, the model is extremely slow and the 1 second delay is very
noticeable despite its reasonable accuracy.

## Discussion
The findings from our exploration presented us with some unexpected turns. Nonetheless, we were able to discern distinct disparities between the performance of GroundingDINO + SAM and YOLO. 

We began the investigation with the presumption that GroundingDINO might struggle a bit with object detection, especially in light of the challenges we faced during its training compared to YOLO. However, what took us by surprise was the real-time detection complications, a by-product of the model's computational demands. As a result, GroundingDINO fell short of the mark when it came to real-time object detection in the arena of first-person shooters on a single GPU setup, primarily due to its slow speed. 

Nevertheless, it's worth mentioning that the model's computation speed could have been influenced by other factors. A glimpse into these can be viewed here: [GitHub Link](https://github.com/BowMonk/valoCV/assets/43303509/75a7e672-ddf6-4640-8ebc-c43cddbb9a60)

In our quest to understand the model better, we shied away from tampering with the numerous intricate operations GroundingDINO undertakes. Tailoring this model to perfection is a time-intensive process. A possible solution could be to run DINO+SAM on multiple GPUs, although that would be a significant commitment for something that other models, like YOLO, can deliver with greater accuracy and speed.
In our effort to maximize efficiency, we optimized the procedure of annotation readings to lessen the workload. Unfortunately, this adjustment didn't impact the overall speed noticeably, as the standalone procedure ran without any noticeable delay.



## Conclusion

In summary, when it comes to real-time object detection tasks, particularly within the scope of first-person shooter scenarios, YOLO emerges as the preeminent choice. Its unique blend of accuracy and speed consistently distinguishes it within the landscape of object detection models. As for hyperparameter tuning, YOLO 5m comes with 18 hyperparameters, which offers flexibility for customization. This level of refinement is currently not accessible with Dino, as it is designed for zero-shot learning.

Nevertheless, Dino's inherent accuracy and versatility cannot be overlooked. Even though it is still in its nascent stages, the model demonstrates substantial potential for future applications in real-time computer vision. The journey of Dino is certainly one to watch as the field of computer vision continues to evolve and innovate.

### Future Directions and Uncompleted Objectives

Our initial goal was to distinguish DINO+SAM from YOLO by utilizing the segmentation masks from SAM to inject variability in the bot's aiming input. The intention was to provide the bot with a less mechanical appearance, an issue that is prevalent among current metadata scraping bots. We envisioned using the boundaries of the segmentation mask as constraints for the noise distribution, which we hypothesized would lead to more realistic bots.

Regrettably, due to the performance of DINO, the feasibility of this study became challenging. Conducting a thorough analysis comparing a basic point-and-click approach with the addition of noise via our proposed method proved to be a time-consuming endeavor.

Despite these challenges, the course of this study led us to conceive an alternative, potentially superior approach that could better leverage all the models discussed. Our proposition entails using GroundingDINO for automated data annotation, which could then be utilized for training YOLO, thereby enhancing object detection performance. Following this, SAM could be adjusted to cooperate with the object detections made by YOLO. It's important to note that while the segmentation masks should remain operative, the compatibility of the text prompt functionality that integrates well with DINO might be compromised in this new setup.

Moving forward, the directions for future work could encompass a deeper examination of DINO's performance and the issues we encountered. Future research should also evaluate the effectiveness of the proposed integration of GroundingDINO, YOLO, and SAM, comparing its performance against the original DINO+SAM approach. Efforts could be made to develop hybrid segmentation techniques, combining the strengths of various models to provide more reliable and effective object detection and segmentation.

Continuing our initial intention, we recommend further exploration into methods of noise introduction to the bot's aiming input to enhance the naturalness of its actions. As the textual prompt functionality may not be possible with the new approach, alternative techniques to emulate this functionality could be an intriguing research direction. Lastly, efforts to optimize the training of YOLO with GroundingDINO's automatic data annotation may result in improved object detection performance and efficiency. Following successful implementation and evaluation, it would be beneficial to investigate the adaptability of this new approach to other use cases or scenarios.


## Tags
`#ComputerVision` `#AI` `#ML` `#YOLO` `#GroundingDINO` `#SAM` `#Valorant` `#Aimbot`

**References**

[1] Redmon, Joseph & Divvala, Santosh & Girshick, Ross & Farhadi, Ali. (2016). You Only Look Once: Unified, Real-Time Object Detection. 779-788. 10.1109/CVPR.2016.91.
