+++
image = "img/portfolio/recursion.png"
showonlyimage = false
date = "2019-08-31T19:44:32+05:30"
title = "Recursion Cellular Image Classification"
draft = false
weight = 10
+++

Disentangling biological signal from experimental noise in cellular images
<!--more-->

![rxrx][1]
[1]: /img/portfolio/rxrx.png
### Introduction
This post presents my solution and lessons learned for Recursion Cellular Image Classification. In this competition, participants have to disentangle experimental noise from real biological signals. Entries will classify images of cells under one of 1,108 different genetic perturbations. The goal of the competition is to help eliminate the noise introduced by technical execution and environmental variation between experiments.

### Lessons learned
#### Running GPU instances on Google Cloud Platform (GCP)
Frankly, the setup took me a while and the tutorial on Google Cloud's office site was not entirely straight forward. I decided to follow the setup on `fast.ai`: https://course.fast.ai/start_gcp.html.

After setting up an instance, I downloaded the data using `CurlWget`, thanks to `ecdrid`. To ensure an instance runs continously even if I shut down my laptop, I resorted to use `tmux`, again under `ecdrid`'s sugesstion.

Connecting to GCP

1. Login to console.cloud.google.com

2. Start an instance, e.g., `instance-p4`

3. Under `SSH`, click on view gcloud command and copy the command line

4. Go to terminal, paste the command


#### Some taken for granted image classification rules

##### On multi GPUs and batch size
I originally worked on the task using Kaggle's kernel but after I received the GCP credits, I switched to GCP and wanted to "speed up" my experiments using multi GPUs. Naively, I turned on 4 GPUs for a instance and chose to increase my batch size by 4 times the original batch size.

To my dismay, the same code did not yield the same results, nothing close. I then learned that if you increase the batch size, you should also increase the learning rate.

##### On image size
Another tip that is often seen in the realm of image classfication is to use smaller image to speed up experiments. I started with image size of 256 and then changed to 320. It was taken for granted that our neural net will be able to handle different sizes of images and I did not question why was this the case. It was a month after the competition that I started to ask such question:

> If we change the size of an image, we did not change anything to the model, what has changed?

Thanks to `ryches`, the answer is because of `Pooling` layers! 

Imagine you have a 512x512 image. Then you pass 4 filters over it and dont add any padding. Now you would have 4 510x510 images for each filter you viewed the original under and you had to cut the edges off because you didnt want to run your filter over white space. Those filters can still be applied to a 224x224 image to make 4 222x222 images.

Now imagine you keep running progressively more filters over that image until it's a very small image but you have a lot of filtered views. so the image has been compressed down to 32x32 and you have 128 filtered views of that 32x32. The 224 x 224 image could have been passed through the same filters to end up with a final 4x4 set of 128 images.

That is when the problem comes. If you just flatten that 32x32x128 and then pass through a dense layer then you will end up with a different number of neurons than the 4x4x128 set. If you do global max pooling or global average pooling though then it will find the average across the entire 32x32 region to make 1 value. then it will just be a 1x128 vector. If you run the same averaging process on a 4x4x128 set then you will still end up with 1x128 vector.

So in the end of the day, if we have 128 filters, a (3, 512, 512) image size until the layer before pooling is (32, 32, 128).  A (3, 224, 224) image size until the layer before pooling is (4, 4, 128). So a pooling layer will ensure the dimension to be 128.

##### On normalization
Since we would like to apply transfer learning to our task, it is advisable to normalize our images with respect to ImageNet's statistics. Unfortunately, this competition has 6 channels and ImageNet only contains 3 channels. Upon a closer analysis, the images are in gray scale, which means we are able to normalize using `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` since images are not "natural". To change the pretrained model's head, the following code was used:
```
with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel_weights, 1)]*6, dim=1)
```

##### On image resolution and model depth
I started to wonder the difference between using large model with small images and small model with large images.

Seeing that most papers reporting results on ImageNet are using images that have been resized to smaller dimensions, does this put smaller models at an unfair disadvantage?

While larger models are better at producing state of the art results when using smaller image sizes, small models can handle larger images giving the same computing resources.

After reading the paper on `EfficientNet`, I came to the conclusion big model works better because of model depth. For a limited amount of GPU memory, we can either go for higher resolution images with small models or lower resolution images with large models. Apparently, we can see this as a trade-off between resolution and model depths. There is a stronger preference to using larger models compared to using higher resolution images. The former gives a significant improvement over the latter.

#### Logging experiments with `wandb`
I find it fairly hard to keep track of all the experiments and chanced upon wandb.com on a YouTube video series called Full Stack Deep Learning. After exploring, I notice how `wandb` is so intuitive and easy to use. It basically logs all my experiments under the config file I have set up.

#### Applying ideas from papers
Here are the papers that worked:

1. Radam optimizer: https://arxiv.org/abs/1908.03265v1
* CNN bag of tricks: https://arxiv.org/abs/1812.01187v2 => label smoothing works, mixed precision did not work
* Deep face recognition: http://arxiv.org/abs/1804.06655v8 => CosFace, ArcFace
* AdamW: https://arxiv.org/abs/1711.05101v3
* SGDR: https://arxiv.org/abs/1608.03983v5
* Random erasing: https://arxiv.org/pdf/1708.04896.pdf

Here are the papers that did not work:

1. Attention-Aware Generalized Mean Pooling for Image Retrieval: https://arxiv.org/abs/1811.00202v2
* Bag of tricks for re-ID: https://arxiv.org/abs/1903.07071v3 => Triplet loss 
* Classification as a strong baseline for metric learning: https://arxiv.org/abs/1811.12649v2 => NormSoftmax low accuracy

Papers that I wish I had time to try:

1. EnsembleNet: https://arxiv.org/abs/1905.09979v1
* Multi sample dropout: https://arxiv.org/abs/1905.09788v2
* CutMix: https://arxiv.org/abs/1905.04899v2
* Ensemble features: https://arxiv.org/abs/1901.05798v1

#### Running bash scripts
Although simple, I was not taught that `.sh` and `.yml` are such convenient setups that allow us to tweak our settings while conducting different experiments, such as tuning for learning rate or loss function.

In my `.sh` file, I was able to run multiple experiments with only a command `deploy.sh`. The file itself contains different arguments for different experiments.

### Solution overview
First I concatenated all 6 channels together and only applied `RandomErasing` as an augmentation strategy to the training data. All images are normalized depending on the cell types.

The best model is the antialias DenseNet-121 pretrained on ImageNet. The head of the model is as follows:

```
self.pooling = AdaptiveConcatPool2d()
self.flatten = Flatten()        
self.bn1 = nn.BatchNorm1d(final_in_features * 2)
self.fc1 = nn.Linear(final_in_features * 2, final_in_features)
self.relu = nn.ReLU(inplace=True)
self.bn2 = nn.BatchNorm1d(final_in_features)   
self.dropout1 = nn.Dropout(p=0.25)
```

CosFace was also successfully applied and managed to achieve better results than `CrossEntropy` and `LabelSmoothingCrossEntropy`. Finally, both sites are summed to predict the label.

We also exploited the leak based on code shared by `nosound`. In the end, we ended up at 64/866 on the private leaderboard. 

### Acknowledgement
I would like to extend my gratitude to Kaggle and Recursion Pharmaceuticals for holding such a fascinating competition. This competition has further reinforced my interest in computer vision and deep learning.
Thank you `pudae` for sharing such an elegant code for his previous competitions on his Github.
Special thanks to team Double Strand, `nosound` and `yuval reina` in offering guidance and tips on training my network and exploiting the leak in the dataset.
Thanks to `phalanx` for sharing his approach in the beginning on the competition.
Thanks to `Leigh` for his starter code.
Thanks to `dohlee` and `grib0ed0v` for sharing their strategies and participating in discussions generously.
Also, the journey would not be possible without my teammate, Chloe Wang. She has been helpful in helping out with the model.