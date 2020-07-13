# This is the notebook I used to create and train the model depolyed on the webpage mention in the README.md file.
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

path = Path('data/Fruit')
path.ls()

# Now it's time to create a data bunch. 
# In this case, I used an 80/20 split for training/validation sets to stay consistent with my experimentation notebook (Fruit Classifier.ipynb).
np.random.seed(42)
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
                                ds_tfms=get_transforms(), size=224, bs=64, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))

# learn is my learning object which I train throughout this notebook. 
# I used ResNet50 as the model in attempt to get better results (relative ro ResNet34).
learn = cnn_learner(data, models.resnet50, metrics=error_rate)

learn.fit_one_cycle(4)

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

# I saved the model after each training session in case it started to overfit. If it did, I could just load a previous model.
learn.save('stage-1')

learn1.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.save('stage-2')

learn.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))
learn.save('stage-3')

learn.fit_one_cycle(2, max_lr=slice(3e-5,5e-4))
learn.save('stage-4')

# After seeing the continual decrease in the error_rate in these training cycles with fewer epochs, 
# I decided to load 'stage-1' and run it for the same amount of epochs, but all at once.
learn.load('stage-1')
learn.fit_one_cycle(6, max_lr=slice(3e-5,5e-4))

# I decided to go with the model I saved as 'stage-4', as it had the lowest final error rate (by a small amount), 
# and the error rate was on a downwards trend in the final epoch.

learn.load('stage-4')
learn.save('Fruit_rs50_good')
learn.load('Fruit_rs50_good')

learn.export()
