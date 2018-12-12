# Intellegent Systems - Project 2
## Aaron Lafitte - A01852530


The purpose of this project was to create an object classifer which could run in real time.
I understand the setup for this is rather difficult and requires a power GPU to run reasonably,
so I have added unit tests that will simply classify images that I have provided.
To demonstrate that I am able to detect objects in real time, I have uploaded a screen recording 
of my roomate's pc doing the object detection with a webcam and a GTX 1070 graphics card. I used my
roomate's computer because I do not have the luxury of having a nice graphics card. Below is an image
of the video that will take you to the video itself if you click on it.

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



