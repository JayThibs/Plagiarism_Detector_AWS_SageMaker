# Plagiarism Detector with AWS SageMaker

This repository contains the code for building and deploying a plagiarism detector on AWS SageMaker.

For this project, I trained a plagiarism detector by looking at similarities in text and training a Random Forest model. 

I also tried training a multi-modal model that makes use of both the text and specific information about the text (tabular data). I was able to setup a model, but due to some SageMaker-specific errors, I decided to move on. That said, I learned a lot about multi-modal models while working on this project and used the [multimodal-transformers](https://github.com/georgian-io/Multimodal-Toolkit) package to do some experiements and train models with text+tabular data (I did this in a Colab notebook).
