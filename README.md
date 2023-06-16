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
of dataset 1 and 2 due to its resolution. However, we approach this differently for both Yolo and GroundingDino with SIEM as we will explain in the next sections.

### YOLO Versions and Variants: Choosing the Right One for Our Project

Navigating through the diverse landscape of object detection algorithms, one cannot miss the dynamism and robustness of YOLO (You Only Look Once)[1]. Known for its speedy real-time inference, YOLO has seen multiple iterations, each carving a niche for itself with unique capabilities and optimizations.

The spotlight often falls on YOLOv3, a common choice among researchers and developers alike. However, the YOLO universe doesn't stop there. The emergence of newer versions like YOLOv7 and YOLOv8 underscores continuous innovations in the pursuit of faster and more efficient real-time object detection[^1^].

For this project, our journey led us to YOLOv5, an intriguing balance between performance and computational resource consumption. But the YOLOv5 story isn't one-size-fits-all, with different variants ('s', 'm', 'l', and 'x') offering a spectrum of speed and accuracy trade-offs[^2^].

- The 's' variant, being the smallest and fastest, tempts with high speed but at the expense of accuracy.
- The 'm' variant, our chosen companion for this endeavor, strikes a harmonious balance between speed and accuracy.
- The 'l' and 'x' variants, on the other end of the spectrum, offer higher accuracy but at the cost of slower inference speed.

The choice of the 'm' model wasn't random[2]. It emerged as the optimal solution considering:

1. **Computational Resources**: The 'm' variant adeptly balances computational efficiency and accuracy. While the 'l' and 'x' could have offered higher accuracy, their larger computational appetite wouldn't fit the bill for our use-case.
2. **Real-Time Performance**: With our application deeply rooted in the fast-paced dynamics of Valorant gameplay, maintaining real-time performance was paramount. The 'm' variant checked this box effectively.
3. **Dataset Complexity**: Our datasets brought with them their own set of challenges, with varying complexities and quality. The 'm' model showcased its robustness in handling diverse image resolutions and noise levels.

Remember, the choice of the model variant significantly hinges on your specific use case and available resources. If your priority is high accuracy and computational resources aren't a constraint, the larger 'l' or 'x' models might just be your ideal match. However, for this project, 'm' proved to be our optimal partner.

## Training and Validation: Harmonizing Datasets and Embracing Diversity

For training YOLO, we utilized three distinct datasets, each with its unique characteristics but adhering to a consistent format. The quality of images within these datasets varied, as did the labels they utilized.

All datasets encompassed `EnemyHead` and `EnemyBody` labels, which were our primary focus for this study. However, they also contained additional labels such as `BoomBot` and other game-specific abilities. These extra labels, although interesting, did not align directly with our current objective.

To achieve a streamlined training process, we undertook the task of re-labeling these datasets. This harmonization allowed us to align the data more closely with our goal. We adopted the following label convention: `['boomybot', 'enemyBody', 'enemyHead', 'smoke', 'splat', 'teammate', 'walledoff enemy']`. This decision was guided by the inner workings of YOLO.

YOLO operates based on class predictions rather than class names, translating our meticulously chosen labels into a plain numerical list: `[0, 1, 2, 3, 4, 5, 6, 7]`. Through our re-labeling effort, we ensured that every class number correctly corresponded to its respective label across all datasets.

One might question our decision to use a wide range of image qualities, rather than solely high-resolution images or curating a pruned selection. The answer lies in the trade-off between quality and diversity. Incorporating images of varying quality level challenged YOLO's ability to interpret images under different conditions. 
This not only enriched our training data but also aimed to enhance YOLO's generalization performance on the test set,
crafting a more robust model ready to tackle different angles one might encounter in Valorant.



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

## Tags
`#ComputerVision` `#AI` `#ML` `#YOLO` `#GroundingDINO` `#SAM` `#Valorant` `#Aimbot`

**References**

[1] Redmon, Joseph & Divvala, Santosh & Girshick, Ross & Farhadi, Ali. (2016). You Only Look Once: Unified, Real-Time Object Detection. 779-788. 10.1109/CVPR.2016.91.
