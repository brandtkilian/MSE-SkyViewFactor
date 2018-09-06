# MSE-SkyViewFactor

This was one of my Master's Degree project, inidividual project of about 100hours of work. The dataset was created by myself and can't be shared due to images rights. The details and results can't be shared as well.

The project consists of computing the Skyviewfactor (SVF) from hemispheric images (taken by an embeded camera in a backpack while walking) using image processing technics and a CNN SegNet as segmentation tools.

This project is part of a big project called CityFeeld developed by the Hepia in Geneva and directed by the team of Peter
Gallinelli and Reto Componovo that target the goal of increase the understanding of the factors
that influence the well-being of people in urban environments

## Image Processing

I've used multiple technics

1. Segment using color: with thresholds, in HSV and RGB)
2. Watershed: with initial markers chosed using color, Otsu tresholding or texture analysis (variance)

## SegNet

The Segnet architecture used in this project is inspired by the work of [pradyu1993](https://github.com/pradyu1993/segnet) that made a Keras implementation of the SegNet proposed in a paper from the Cambridge University that you can find [there](http://arxiv.org/pdf/1511.00561v2.pdf).

My final trained model recognize 3 classes (4 if we take into account the "void" classes that are black pixels due to the fisheye) with an accuracy of about 95%:
* Buildings
* Sky
* Vegetation

Since the dataset has been handmade for this project, I used some Data Augmentation technics.

## Pipeline

![Pipeline](https://raw.githubusercontent.com/brandtkilian/MSE-SkyViewFactor/master/figures/pipeline.png)

1. Correct projection and keep only the 180° (half sphere) above horizontal from images. The projection is corrected with a calibration image on which a graduated perfect arc circle is taken on photograph, the radius function is then interpolated.
2. Segment image using Image Processing algorithms or trained SegNet (SegNet is much more better)
3. Compute Sky View Factor

![Sky view factor computing](https://raw.githubusercontent.com/brandtkilian/MSE-SkyViewFactor/master/figures/svf_algo.png)

The sky view factor is computed with an iterative algorithm (a kind of integral computation) knowing the angle of each circles. The algorithm is presented in the paper [Holmer Björn, A simple operative method for determination of sky
view factors in complex urban canyons from fisheye photographs, January 1992. Web. 04
June 2017](http://bit.ly/2rzb2zT)

## A SegNet segmented image with the three classes segmented (overlay)

Just an image to show the segmentation results by SegNet
![Segmentation results](https://raw.githubusercontent.com/brandtkilian/MSE-SkyViewFactor/master/figures/segmentationoverlay.jpg)




