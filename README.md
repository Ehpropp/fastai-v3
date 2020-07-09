# Fruit Classifier Web App Repository

This repo is the code base for the web app version of a deep learning model I created using the [fast.ai](https://www.fast.ai) library. Upload an image of any of the listed fruits on the page and let it tell you what fruit you uploaded. Try it out and let me know how it goes!

The web app can be accessed [here](https://eli-fruit-classifier.onrender.com).

I trained the model using the ResNet50 model with a dataset I created from Google images (taught in the fastai course linked below). I used around 300 images per category, with an 80/20 split of training/validation sets. No test set was created at the moment as I continue the course mentioned below, although I may create one at a later date. I cleaned a large portion of the data myself using a tool from the fast.ai library, but not all of it. I may clean it further later on.

The notebook file titled 'Fruits Good.ipynb' is the notebook used to create the working model after I cleaned the data and tried different approaches. The notebook where I cleaned setup everything will be uploaded soon.

Check out the [Practical Deep Learning for Coders](https://course.fast.ai/) course by fast.ai that taught me how to create this!

This repo was forked from the [fast.ai](https://www.fast.ai) starter repo for deploying their models on [Render](https://render.com). The original repo from fast.ai can be found [here](https://github.com/render-examples/fastai-v3).

I did not write the code base for the html or server.py file; I only made the necessary changes for my project.
