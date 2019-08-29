+++
image = "img/portfolio/kmx.jpg"
showonlyimage = false
date = "2016-11-05T19:57:40+05:30"
title = "CarMax Sales Inventory"
draft = false
weight = 7
+++

An exploratory data analysis and sales forecase for CarMax
<!--more-->

#### Introduction
Here, I was given a data science challenge: to perform meaningful exploratory data analysis for a CarMax dataset scrapped from CarMax.

My idea was the final day a car is listed on CarMax is the day it was sold.

To test the idea, the following hypothesis is formed: 171,076 cars are sold between March and May 2016. The figure is based on page 26 of the 10Q by CarMax.

I shall find out if this is true by creating a dataframe only with entries from March to May 2016.
```
tail_df = df.groupby('url').tail(1)
sum(tail_df['price'])
```

The total sales from March to May is 3,350,894,231 USD while the 10Q reported 3,428,974 USD.

#### Exploratory data analysis
We have 1,250,006 unique entries in our dataset, collected for 599 days across cars sold in 308 regions.

The average days it takes to sell a car is 26 while the median is 17.

Out of the 40 make, which is the most popular?

![kmx-make][1]
[1]: /img/portfolio/kmx-make.png

Cars having price ranging from 11,000 USD to 33,000 USD are most popular.

![kmx-sold][2]
[2]: /img/portfolio/kmx-sold.png

10,339 cars were sold on 2017-01-24, which is unusual.


Which region has the best sales?
```
cars_bin_df['region'].value_counts()[:10]

Norcross                  22334
Ontario                   18291
Texas Stadium (Irving)    17117
Fredericksburg            17016
Columbia                  17002
Lancaster                 16759
Hartford                  16264
Austin North              15748
Arlington/Ft. Worth       14999
Oxnard                    14865
```
Norcross seems to have the best sales, followed by Ontario.

#### Can we build a model to predict how long does it take to sell a car?
Let us split our dataset in terms of time, having training data from 12-2015 to 08-2016 and test data from 9-2016 to 11-2016. These dates align with CarMax's 10-Q. We then try to estimate the sales for our test data.

Here, I attempted to apply Entity Embedding using `fastai`. Here is a snippet:

```
learn = tabular_learner(data, layers=[1000,500], 
                        ps=[0.001,0.01], 
                        emb_drop=0.04, 
                        y_range=y_range, 
                        metrics=mse)
learn.model

TabularModel(
  (embeds): ModuleList(
    (0): Embedding(41, 13)
    (1): Embedding(2586, 130)
    (2): Embedding(311, 40)
    (3): Embedding(15, 7)
  )
  (emb_drop): Dropout(p=0.04)
  (bn_cont): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=193, out_features=1000, bias=True)
    (1): ReLU(inplace)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.001)
    (4): Linear(in_features=1000, out_features=500, bias=True)
    (5): ReLU(inplace)
    (6): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.01)
    (8): Linear(in_features=500, out_features=1, bias=True)
  )
)
```
The model only predicted 128,530 cars would be sold in the 3 months, quite a good estimation compared to 156,789 (from page 22 of 10-Q).