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


#### Datasets

#### Yolo
Do note that all these datasets contain different labels with different quality of images. Here are some random examples of all 3 datasets that was used to train Yolo

<h2 align="center">Dataset 1</h2>
<div align="center">
<table>
  <tr>
    <td align="center"><a href="#"><img src="images/dataset1/11.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/12.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/13.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><a href="#"><img src="images/dataset1/14.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/15.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/16.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><a href="#"><img src="images/dataset1/17.jpg" width="200px;" alt="Image7"/><br /><sub><b>Image 7</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/18.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></a></td>
    <td align="center"><a href="#"><img src="images/dataset1/19.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></a></td>
  </tr>
</table>
</div>

and 
This setup enables SAM and GroundingDINO to identify and target areas with higher accuracy due to color labeling.  


<br>
<br>

<h2 align="center">Dataset 2</h2>


<div align="center">
<table>
  <tr>
    <td align="center"><img src="images/dataset2/21.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></a></td>
    <td align="center"><img src="images/dataset2/22.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></a></td>
    <td align="center"><img src="images/dataset2/23.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset2/24.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></a></td>
    <td align="center"><img src="images/dataset2/25.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></a></td>
    <td align="center"><img src="images/dataset2/26.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset2/27.jpg" width="200px" alt="Image7"/><br /><sub><b>Image 7</b></sub></a></td>
    <td align="center"><img src="images/dataset2/28.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></a></td>
    <td align="center"><img src="images/dataset2/29.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></a></td>
  </tr>

</table>
</div>

<br>
<br>

<h2 align="center">Dataset 3</h2>



<div align="center">
<table>
  <tr>
    <td align="center"><img src="images/dataset3/31.jpg" width="200px;" alt="Image1"/><br /><sub><b>Image 1</b></sub></a></td>
    <td align="center"><img src="images/dataset3/32.jpg" width="200px;" alt="Image2"/><br /><sub><b>Image 2</b></sub></a></td>
    <td align="center"><img src="images/dataset3/33.jpg" width="200px;" alt="Image3"/><br /><sub><b>Image 3</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset3/34.jpg" width="200px;" alt="Image4"/><br /><sub><b>Image 4</b></sub></a></td>
    <td align="center"><img src="images/dataset3/35.jpg" width="200px;" alt="Image5"/><br /><sub><b>Image 5</b></sub></a></td>
    <td align="center"><img src="images/dataset3/36.jpg" width="200px;" alt="Image6"/><br /><sub><b>Image 6</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><img src="images/dataset3/37.jpg" width="200px;" alt="Image7"/><br /><sub><b>Image 7</b></sub></a></td>
    <td align="center"><img src="images/dataset3/38.jpg" width="200px;" alt="Image8"/><br /><sub><b>Image 8</b></sub></a></td>
    <td align="center"><img src="images/dataset3/39.jpg" width="200px;" alt="Image9"/><br /><sub><b>Image 9</b></sub></a></td>
  </tr>

</table>
</div>

It is apparent that dataset 1 has high quality images while dataset 2 and 3 has low quality in-game images. One can decide to remove some of the images
of dataset 1 and 2 due to its resolution. However, we approach this differently for both Yolo and GroundingDino with SIEM.

<h2 align="center">Dataset 4</h2>

The last dataset is simply a video where we test our real-time detection.
In the video, a streamer (TenZ) is practicing in the shooting range found in the game, where enemies spawn one after another in quick succession and they have to be shot before they dissappear.

<p align="center">
<img src='pics/video_example.jpg' width='600'>
<p >

To make our evaluation easier, we annotated all frames where an enemy appears with dots centering on the head and body of the enemy, as can be shown here:

<p align="center">
<img src='pics/shooting annotation.png' width='600'>
<p >

At the same time, during detection moments, to reduce the clutter of detections on the screen, we whited out the camera and ad banner found in pictures.

How we use these points, will be explained in the online evaluation section.


### Yolo

### GroundingDINO and SAM

#### Description

#### Annotations
The annotations of the datasets we used are not very compatible with GroundingDINO and SAM. The annotations are small boxes which fit well with the detection style of YOLO and are seemingly annotated with that in mind.
However, we had trouble getting GroundingDINO to get a similar detection style, simply because it does not use the same grid system that YOLO uses, and the textual prompts trying to isolate some areas do not work well all the time.

The annotations were specifically made for two labels: "Head" and "Person".

To make sure that the datasets were useful for both sides, we manually annotated the datasets, except for the third one as it was too noisy to make consistent decisions on annotating.
For this annotation procedure we used labelme (the package can be installed through pip), and manually annotated boxes that would fit GroundingDINOs detection style.
Some examples of this annotation can be seen below:

<p align="center">
<img src='pics/annotation_example_1.png' width='300'>
<img src='pics/annotation_example_2.png' width='300'>
<p >

<p align="center">
<img src='pics/annotation_example_3.png' width='300'>
<img src='pics/annotation_example_4.png' width='300'>
<p >

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
This search was done using different combinations of box and text thresholds, with a range from 0.05 to 0.45 (thresolds higher than 0.45 were eliminating too many clear cases)
In the end, the best performing cases were around 0.4 to 0.45, however there were many more cases with no detections.
So instead we settled on 0.35 for both with an MAP of 0.803, as it detected nearly all cases.

<div style="position:relative;width:fit-content;height:fit-content;">
            <a style="position:absolute;top:20px;right:1rem;opacity:0.8;" href="https://clipchamp.com/watch/mpzngBMAgzz?utm_source=embed&utm_medium=embed&utm_campaign=watch">
                <img style="height:22px;" src="https://clipchamp.com/e.svg" alt="Made with Clipchamp" />
            </a>
            <iframe allow="autoplay;" allowfullscreen style="border:none" src="https://clipchamp.com/watch/mpzngBMAgzz/embed" width="640" height="360"></iframe>
        </div>

### Offline Evaluation

The key to any comparative study lies in its evaluation metrics. We use Mean Average Precision (MAP), Precision, Recall, and F1 score as our key metrics to evaluate the performance of the models:

- **Precision** is the ratio of correctly predicted positive observations to the total predicted positives.
- **Recall** (Sensitivity) calculates the ratio of correctly predicted positive observations to the all observations in actual class.
- **F1 Score** is the weighted average of Precision and Recall. It tries to find the balance between precision and recall.
- **MAP** (Mean Average Precision) is used in information retrieval to measure the effectiveness of the model in terms of precision of retrieval over a range of recall values.

We run these evaluations both offline and online, with online evaluations conducted on live game feed to measure the inference time and reaction to dynamic movements. 

### Expected Outcomes

Through this experiment, we aim to answer the following questions:

- How does YOLO compare to SAM+GroundingDINO for general metrics like MAP, recall, precision, F1 score?
- What hyperparameters need to be tuned between the two models?
- How well does the model react to dynamic movements in the online game feed?
- Can SAM+GroundingDINO outperform YOLO in a dynamic first-person shooter game like Valorant?

Our study aims to shed light on the applications of advanced computer vision techniques in gaming, specifically first-person shooters. By comparing different object detection architectures, we hope to contribute to the ongoing development of fair and ethical gaming practices.

### Results

#### GroundingDINO and SAM

The results for GroundingDINO and SAM are with respect to their offline performance on dataset 1 + 2, and also the 
online performance on dataset 4.

##### Offline performance

Table I

|    Class    |   MAP   |   AP    | Recall  | Precision | F1_Score |
|:-----------:|:-------:|:-------:|:-------:|:---------:|:--------:|
|    Head     | 0.62811 | 0.32946 | 0.27487 |  0.67112  | 0.39001  |
| Person/Body | 0.62811 | 0.92677 | 0.49863 |  0.97919  | 0.66077  |

At first glance, the most noticeable thing is the disparity between the AP (Average Precision) between the Head and Person/Body classes.
DINO was much better at detecting the general body, which is understandable as the head has less detail and is generally
harder to detect due to the smaller amount of pixels.
The model in general has low recall but higher precision, with overall not a great performance, much less than we expected.

##### Online performance

When it comes to online performance, our main measure is delay during the real-time detection.

Sadly, GroundingDINO and SAM performed quite poorly.
The average delay of detecting object per frame was around 1.11202 seconds. That is a very long time, even for a 30 FPS video.
In a real gaming environment, FPS in shooter games is quite important, and having a bot that can shoot every 65 frames or so
if the gaming is running at 60 FPS, is quite bad.

Despite this, when it comes to accuracy it performed similarly to the offline setting in general accuracy, having difficulty 
with the Head class and consistently detecting the Person/Body class with a slight boost, due to less distance from the 
camera to the enemies in the shooting range.

The exact values for MAP, AP, Recall, Precision and F1_Score were not calculated due to the video not having any annotations
for the boxes (the rest of the annotation took a considerable amount of time). 

## Tags
`#ComputerVision` `#AI` `#ML` `#YOLO` `#GroundingDINO` `#SAM` `#Valorant` `#Aimbot`

**References**

[1] Redmon, Joseph & Divvala, Santosh & Girshick, Ross & Farhadi, Ali. (2016). You Only Look Once: Unified, Real-Time Object Detection. 779-788. 10.1109/CVPR.2016.91.
