# filter out bias from weight decay
# decaying learning rate with cosine schedule
# half-precision Adam statistics
# half-precision stochastically rounded text encoder weights were used

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

import clip

BATCH_SIZE = 32768
# define your own dataloader
train_dataloader = DataLoader(..., batch_size=BATCH_SIZE)


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# if using GPU then use mixed precision training
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# must set jit equals False for training
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
if device == "cpu":
    model.float()
else:
    # actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model)

# set loss function with respect to both image and text
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

# set max epoch number
EPOCH = 32
for epoch in range(EPOCH):
    for batch in train_dataloader:
		# initialize backprop gradient tensor of optimizer 
        optimizer.zero_grad()
        # list_images equals list of image in numpy array(np.uint8), or list of PIL images
        list_image, list_txt = batch
        # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        images = torch.stack([preprocess(Image.fromarray(img)) for img in list_image], dim=0).to(device)
		# tokenize text
        texts = clip.tokenize(list_txt).to(device)
		# CLIP.foward() calculates cosine similarity as logits
        logits_per_image, logits_per_text = model(images, texts)
		# set ground truth to calculate loss
        ground_truth = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
		# set total loss as the mean of image loss and text loss
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
			# training with GPUs
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
