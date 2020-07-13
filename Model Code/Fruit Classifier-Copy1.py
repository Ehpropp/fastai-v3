# This is the original notebook used to create my data set and experiment with training. 
# For the notebook used to create the final model exported to the webpage, look at the 'Fruits-Good.ipynb' or 'Fruits_Good.py' file.
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.metrics import error_rate

folder = 'Apple'
file = 'urls_apple.csv'

folder = 'Banana'
file = 'urls_banana.csv'

folder = 'Blackberry'
file = 'urls_blackberry.csv'

folder = 'Blueberry'
file = 'urls_blueberry.csv'

folder = 'Lemon'
file = 'urls_lemon.csv'

folder = 'Lime'
file = 'urls_lime.csv'

folder = 'Mango'
file = 'urls_mango.csv'

folder = 'Orange'
file = 'urls_orange.csv'

folder = 'Pear'
file = 'urls_pear.csv'

folder = 'Raspberry'
file = 'urls_raspberry.csv'

folder = 'Strawberry'
file = 'urls_strawberry.csv'

folder = 'Tomato'
file = 'urls_tomato.csv'

path = Path('data/Fruit')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
path.ls()

classes = ['Apple', 'Banana', 'Blackberry', 'Blueberry', 'Lemon', 'Lime', 'Mango', 'Orange', 
          'Pear', 'Raspberry', 'Strawberry', 'Tomato']


# To download the images from Google images, I used a script shown in the fast.ai course. 
# The script saves the urls for the all loaded images locally. 
# I then uploaded them to the appropriate directory and then used their function to extract the images from the saved urls for each category. 
#Here is the script:
# urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
download_images(path/file, dest, max_pics=300)

for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
          
# Now it's time to create a data bunch, which is what fast.ai calls their data sets. 
# The parameter 'valid_pct' is the percentage of the images in my dataset which will act as the validation set. 
# In this case, I used an 80/20 split for training/validation sets.
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

data.classes

data.show_batch(rows=3, figsize=(7,8))

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

# I initially attempted to train using the ResNet34 framework.
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')

# The unfreeze() function essentially allows the rest of the layers in the model to be trained. 
# Without unfreezing, the fast.ai library only trains the last layers that gets adds to the framework.

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(2, max_lr=slice(3e-6,5e-5))

learn.save('stage-2')

learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Some of the error has come from incorrectly labeled images or images that aren't actually one of the fruits, 
# so the model is probably doing better than it says.
interp.plot_top_losses(9, figsize=(15,11))

# The ImageCleaner widget created by fast.ai is what I used to clean the dataset. 
# The widget will not show unless I'm in the process of cleaning, but it easily allows me to delete or rename images, or leave them as is. 

from fastai.widgets import *
db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch())

learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
learn_cln.load('stage-2');

ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, path)

np.random.seed(42)
data1 = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
                                ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

data1.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn1 = cnn_learner(data1, models.resnet34, metrics=error_rate)

learn1.fit_one_cycle(4)
learn1.save('stage-1')

learn1.unfreeze()
learn.lr_find()
learn1.recorder.plot()

learn1.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn1.save('stage-2')
learn1.load('stage-2');

interp1 = ClassificationInterpretation.from_learner(learn)
interp1.plot_confusion_matrix()
learn1.save('stage-2')

learn1.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn1.save('stage-3')

learn1.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn1.load('stage-3')

db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')
                   .split_none()
                   .label_from_df()
                   .transform(get_transforms(), size=224)
                   .databunch())

learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
learn_cln.load('stage-3');

ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, path)

path = Path('data/Fruit')

np.random.seed(42)
data2 = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
                                ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

# I trained the model again with the exact same training and validation sets with ResNet34, now starting with a mostly cleaned dataset.
learn2 = cnn_learner(data2, models.resnet34, metrics=error_rate)
learn2.fit_one_cycle(4)

learn2.unfreeze()
learn2.lr_find()
learn2.recorder.plot()

learn2.fit_one_cycle(2, max_lr=slice(3e-6,3e-5))
learn2.fit_one_cycle(2, max_lr=slice(6e-4,8e-3))

# I then decided to try training with ResNet50 it to try and get better results. This is with the already cleaned data set.
learn3 = cnn_learner(data2, models.resnet50, metrics=error_rate)
learn3.fit_one_cycle(4)

learn3.save('stage-1-3')

learn3.unfreeze()
learn3.lr_find()
learn3.recorder.plot()

learn3.fit_one_cycle(3, max_lr=slice(6e-5,6e-4))
learn3.save('stage-2-3')

learn3.fit_one_cycle(1, max_lr=slice(6e-5,6e-4))
learn3.save('stage-3-3')

learn3.load('stage-3-3')
