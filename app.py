# %%
"""
# The following cell connects Kaggle to the internet for successful running of the code
"""

# %%
#NB: Kaggle requires phone verification to use the internet or a GPU. If you haven't done that yet, the cell below will fail
#    This code is only here to check that your internet is enabled. It doesn't do anything else.
#    Here's a help thread on getting your phone number verified: https://www.kaggle.com/product-feedback/135367

import socket,warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")

# %%
"""
# Including all necessary packages and libraries
"""

# %%
pip install -Uqq fastai
pip install gradio

from fastai.vision.all import *
import gradio as gr

pip install duckduckgo_search

from duckduckgo_search import ddg_images
from fastcore.all import *

from fastdownload import download_url

# %%
"""
# This cell consists of the function that gets image from the internet using DuckDuckGo library, which imports images from the internet
"""

# %%
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

# %%
"""
# This cell displays the URL thats been searched and provided of the image searched
"""

# %%
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('black person', max_images=1)
urls[0]

# %%
"""
# This cell downloads and displays the image thats been searched
"""

# %%
dest = 'black-person.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(192,192)

# %%
"""
# This cell downloads the second image thats been searched for classifying the two images respectively
"""

# %%
download_url(search_images('white person', max_images=1)[0], 'white-person.jpg', show_progress=False)
Image.open('white-person.jpg').to_thumb(192,192)

# %%
"""
# This cell searches for a random image for testing purposes
"""

# %%
download_url(search_images('random', max_images=1)[0], 'dunno.jpg', show_progress=False)
Image.open('dunno.jpg').to_thumb(192,192)

# %%
"""
# This cell searches different images with the subject being the same in different scenarios, to train the model better and for better results
"""

# %%
searches = 'white person','black person'
path = Path('black_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} photo in sun'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} photo in shade'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# %%
"""
# This cell removes the error images that may corrupt the found dataset
"""

# %%
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# %%
"""
# This block loads the images from the dataset acquired and further classifies it into their categories respectively
"""

# %%
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

# %%
"""
# This cell trains on the photos taken from the internet and calculates the metrics of the model and the dataset its being trained on
"""

# %%
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# %%
"""
# This cell checks how well the model will run if its provided with any other input related to the image classication database
"""

# %%
is_black,_,probs = learn.predict(PILImage.create('black-person.jpg'))
print(f"This is a: {is_black}.")
print(f"Probability it's black person: {probs[0]:.4f}")

# %%
"""
# This cell exports the trained dataset
"""

# %%
learn.export('racialModel.pkl')

# %%
learn = load_learner('racialModel.pkl')

# %%
download_url(search_images('african male', max_images=1)[0], 'test.jpg', show_progress=False)
Image.open('test.jpg').to_thumb(192,192)

# %%
learn.predict('test.jpg')

# %%
categories = ('Black Person', 'White Person')

def classify_img(img):
    pred,pred_idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# %%
classify_img('test.jpg')

# %%
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['white-person.jpg','black-person.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)