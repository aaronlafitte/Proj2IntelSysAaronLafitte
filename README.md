# Intellegent Systems - Project 2
## Aaron Lafitte - A01852530


The purpose of this project was to create an object classifer which could run in real time.
I understand the setup for this is rather difficult and requires a powerful GPU to run reasonably,
so I have added unit tests that will simply classify images that I have provided.
To demonstrate that I am able to detect objects in real time, I have uploaded a screen recording 
of my roomate's pc doing the object detection with a webcam and a GTX 1070 graphics card. I used my
roomate's computer because I do not have the luxury of having a nice graphics card. 

**Below is an image of the video that will take you to the video itself if you click on it.**

[![Project 2](http://img.youtube.com/vi/xPUh7N2N0bo/0.jpg)](http://www.youtube.com/watch?v=xPUh7N2N0bo "Project 2")

Now that we have seen the classifier run in real time, let's setup your enviorment to run the unit_tests.py.

## Setting up your environment 

I did this entire project using a Conda virutal enivroment so that my packages are all the same version.
https://www.anaconda.com/download/

```
Anaconda 5.3.1
Python 3.7
```
Once you have that installed, open the Anaconda navigator and go the **Environments** tab on the left.

![env](https://user-images.githubusercontent.com/37847947/49839010-60f79880-fd6a-11e8-85ec-3f2ab2a7fabd.JPG)

In the bottom left hand corner, we can see the option to create a new virtual environment. 
Let's create new environment that uses **Python 3.6**. You can name it whatever you'd like. 
After a short bit, the environment wil be ready to go. Click the play button associated with your 
newly created enviornment and select the **Open Terminal** option. 

From the terminal, navigate to project directory using the **cd** command. Mine is on my C:\ so my
command looks like this:
![cd](https://user-images.githubusercontent.com/37847947/49839426-39a1cb00-fd6c-11e8-98c5-2f792334dc3f.JPG)

From here we will excute the following commands:
```
conda install -c anaconda protobuf
```
This will be used to complie tensorflows object detection proto files.
Now run this command:
```
pip install -r required_packages.txt
```
This will just verify that you have the correct packages to run unit_test.py Now we need to compile the proto files using the following command:
```
protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto
```
Near the end. Now run these last two commands:
```
python setup.py build
```
```
python setup.py install
```
Alright! You've done it! Now we run it.

## Running unit_tests.py

All you need to do now is run the command :
```
python unit_tests.py
```

This python script will take take each image inside of the unit_test_images folder and run them through the object detector.
All of these images are from my testing data set, which can be found in data/test.

The expected output should look something like this:
![results](https://user-images.githubusercontent.com/37847947/49842988-d23f4780-fd7a-11e8-8186-a48742419b02.JPG)

## Findings
Real time object detection requires some beefy graphics cards. I was pretty pleased on the outcome of this project. If I could go back I would defienetley need to add to the data set heavily. I also found it funny that the King card confused the model the most. There must be futures like hair that are similar enough to Jacks, Queens, and Kings that makes it confused. I would also be curious to know what adding more numbered cards would do to the model. I feel that the numbered cards are enough alike that the model would have a hard time getting them all right. 

I trained the network using tensorflow's train.py, which is found in the utils folder. The file takes several arguments such as the location of the training and testing data, as well as the inital model used. Tensorflow provides several different models that have been trained to detect objects in general which you can use as a starting point. I used the faster_rcnn_inception_v2_pets.config model to start the initial training. Training took about 6 hours using a 1070 GTX and it appears to be getting pretty good results. If I would have provided a real data set I beleive this would have taken days to train, but would most likely be very accurate. 

## Environment
```
Windows 10
Anaconda 5.3.1
Python 3.6
tensorflow 1.12
numpy 1.15.4
opencv-python 3.4.4.19
Pillow 5.3.0
matplotlib 3.0.2
Cython 0.29.1
Logitec Web Cam
GTX 1070 Graphics Card

```

### Credit

1. I used a lot of this code as an example to follow, as did both of the other links.
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
2. This link really helped code my webcam feed. https://github.com/datitran/object_detector_app/
3. This code was very helpful with demonstatring how to label data as well as providing the data set that I trained on. https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10



