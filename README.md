# Synapse Annotator
A deep neural network for synapse detection

Morgan Neuwirth - Harvard Medical School Regehr Lab, Summer Internship 2019

## Github

- git clone https://github.com/morganneuwirth/synapseAnnotator.git


## Python Environment

- requires:
    - Python 3.7
    - Anaconda (https://www.anaconda.com/distribution/)
    - pytorch (conda install -c pytorch pytorch)
    - skimage (conda install -c conda-forge scikit-image)
    - oiffile (pip install oiffile)

## Normalization
##### /notebooks/normalization

- find mean and mean values of pixel intensity of each channel across every z-slice
- save values to scale images for network

## Annotation
##### /notebooks/annotation

- load OIB file and randomly select a z-slice to annotate
- choose variety of images
- annotation window will pop up
    - left click to annotate synapses
    - center click to remove previous point
    - right click to end annotation
- automatically scales and slices images into four smaller images
- saves image name, synapse coordinates, images, and masks of annotated synapses for a train/test set

## Training
##### /notebooks/trainNetwork

- import and divide test and training images
- set loss function ("softdice"), learning rate (1e-4), batch size (2), number of epochs (1200), and decay schedule (50,0.98)
- set device as torch.device('cuda') to decrease training time
- run loop to generate 10 networks
- visually analyze train/test error graph
- compare test images and predicted synapses to assess network quality
- save networks (torch.save(network, 'network_name')) to use to run program


## Running
##### /notebooks/runProgram

- import saved networks and normalization factors
- loop through every image and each z-slice
- appends the predicted image as an additional channel
- exports original OIB file with added channel as a TIF file

## Acknowledgements

- this work is inspired by DoGNet: A deep architecture for synapse detection in multiplexed fluorescence images
- this code utilizes U-Net: Convolutional Networks for Biomedical Image Segmentation
