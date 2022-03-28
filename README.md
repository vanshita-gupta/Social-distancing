# Social-Distancing-Analyzer-and-Mask-Detector-AI-Tool

## PROJECT OUTLINE
To enforce social distancing protocol in public spaces and workplaces, this project intends to build a Social Distance Analyzer AI Tool that can monitor if individuals are keeping a safe distance from each other by analysing real-time video feeds from the camera. It has the ability to identify where each individual is in real-time and produce a bounding box that turns red if the distance between two persons is too close. It can also determine whether a person is wearing a mask or not by utilising a pre-trained CNN-based model.
This may be utilised by concerned authorities to analyse people's movements and inform them if the situation worsens.

## PROJECT WORKFLOW
★ Person detection using YoloV3 and getting bounding boxes in real-time

★ Face detection i.e Extracting Region of Interest (face) in each detected person’s bounding box

★ Face Mask Classifier (MobileNetV2 Architecture) to identify whether person is wearing mask or not

★ Social distancing measurement i.e computing the pairwise distances between all detected people and based on these distances, check if any two people are less than N pixels apart.

★ Display output status in the frame

## VIDEO OF WORKING MODEL
[Project Demo](https://drive.google.com/drive/folders/1o4UNb4sAlY5rg6cVrDR-OFpM0FjOmmzA)


