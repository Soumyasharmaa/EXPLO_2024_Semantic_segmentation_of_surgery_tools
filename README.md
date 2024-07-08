# Semantic Segmentation of Endoscopic Surgery Equipments

This repository presents semantic segmentation models designed for the identification and labeling of instruments in robotic-assisted surgeries. The objective is to accurately classify each pixel within the surgical scene according to its corresponding instrument class. Semantic segmentation of surgical instruments serves as a fundamental task in the development of robotic-assisted surgery systems, facilitating instrument tracking, pose and surgical phase prediction, and more. By developing robust methods for semantic segmentation of surgical instruments, this project aims to drive progress across multiple domains of study.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Presentation](#presentation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Robotic-assisted surgeries are revolutionizing the medical field by providing surgeons with enhanced precision and control. A critical aspect of these systems is the ability to accurately identify and track surgical instruments in real-time. This project focuses on semantic segmentation of endoscopic surgery equipments, a task that involves classifying each pixel in a surgical image into predefined categories.

## Dataset

The dataset used in this project is from the Robotic Scene Segmentation Sub-Challenge of Endovis 2018.

- **Data Collection**: The dataset comprises 19 videos of surgeries, divided into 15 training sets and 4 test sets. Each sequence originates from a single porcine training procedure recorded on da Vinci X or Xi systems, utilizing specialized hardware. The frames were manually filtered until each sequence contained 300 frames.

- **Data Annotation**: A team of trained professionals annotated the data using specialized software, generating boundaries around each semantic class. Quality control measures were implemented throughout the annotation process to ensure consistent labeling.

[Dataset Link](https://drive.google.com/drive/folders/19TncuD9YB3AFbTguFHEDCe6gC9BgDMcL?usp=drive_link)

## Methodology

The semantic segmentation task was approached using the U-Net architecture, a popular model for biomedical image segmentation. The following techniques were employed to optimize model performance:

- **Loss Function**: Jaccard distance loss
- **Optimization**: Early stopping, learning rate reduction, and model checkpointing

## Results

The model achieved an Intersection over Union (IoU) of 0.8809 on the test dataset, demonstrating its effectiveness in accurately segmenting surgical instruments.

## Presentation

For a detailed overview of the project, refer to the [Canva presentation](https://www.canva.com/design/DAGDH_wXFDI/bT9OrZmSc4izFdwMqv31aQ/view?utm_content=DAGDH_wXFDI&utm_campaign=designshare&utm_medium=link&utm_source=editor).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was completed under the supervision of Dr. R. K. Pandey, Head of Department at IIT (BHU), Varanasi.

