
# coding: utf-8

# This is the notebook I used to create and train the model depolyed on the webpage mention in the README.md file. To see how I created my data set and experimented, look at the 'Fruit Classifier.ipynb' file. I explain certain functions and decisions in more detail there.

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[14]:


bs = 64


# In[6]:


path = Path('data/Fruit')


# In[7]:


path.ls()


# Now it's time to create a data bunch. In this case, I used an 80/20 split for training/validation sets to stay consistent with my experimentation notebook (Fruit Classifier.ipynb).

# In[15]:


np.random.seed(42)
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
                                ds_tfms=get_transforms(), size=224, bs=64, num_workers=4).normalize(imagenet_stats)


# In[16]:


data.show_batch(rows=3, figsize=(7,8))


# learn1 is my learning object which I train throughout this notebook. I used ResNet50 as the model in attempt to get better results (relative ro ResNet34).

# In[17]:


learn1 = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[18]:


learn1.fit_one_cycle(4)


# In[19]:


learn1.unfreeze()


# In[20]:


learn1.lr_find()


# In[21]:


learn1.recorder.plot()


# I saved the model after each training session in case it started to overfit. If it did, I could just load a previous model.

# In[22]:


learn1.save('stage-1')


# In[23]:


learn1.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))


# In[24]:


interp = ClassificationInterpretation.from_learner(learn1)


# In[26]:


interp.plot_confusion_matrix()


# In[27]:


learn1.save('stage-2')


# In[28]:


learn1.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))


# In[29]:


learn1.save('stage-3')


# In[30]:


learn1.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))


# In[32]:


learn1.save('stage-4')


# After seeing the continual decrease in the error_rate in these training cycles with fewer epochs, I decided to load 'stage-1' and run it for the same amount of epochs, but all at once.

# In[ ]:


learn1.load('stage-1')


# In[34]:


learn1.fit_one_cycle(6, max_lr=slice(3e-5,5e-4))


# I decided to go with the model I saved as 'stage-4', as it had the lowest final error rate (by a small amount), with the error rate being on a downwards trend in the final epoch.

# In[ ]:


learn1.load('stage-4')


# In[36]:


learn1.save('Fruit_rs50_good')


# In[ ]:


learn1.load('Fruit_rs50_good')


# In[38]:


learn1.export()

