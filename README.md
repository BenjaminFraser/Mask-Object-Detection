# Mask-Object-Detection


## Introduction

A project that develops a Mask Object detection deep neural network model.

For object detection model development, a Custom pre-trained Faster-RCNN architecture was adapted and fine-tuned to a custom face-mask dataset. Refer to the notebooks within this repository for all code and process(es) applied for this.


## Example predictions

Some basic examples of applying the trained model to new inputs are shown below:


![Example Predictions 1](examples/prediction_example_1.png?raw=True "Example of model inference on unseen examples")


![Example Predictions 2](examples/prediction_example_2.png?raw=True "Example of model inference on unseen examples")


![Example Predictions 3](examples/prediction_example_3.png?raw=True "Example of model inference on unseen examples")


As shown, the model can adapt to a range of scenes, both simple and complex. It also works surprisingly well on complex scenes with high numbers of people, which make the final trained model useful for real applications.