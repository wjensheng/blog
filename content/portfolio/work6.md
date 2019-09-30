+++
draft = false
image = "img/portfolio/ey.jpg"
date = "2016-11-05T19:56:17+05:30"
title = "Ernst & Young"
showonlyimage = false
weight = 4
+++

Predict location of mobile device given historical trajectory data
<!--more-->

#### Introduction

This post presents the 24th place solution globally (and 1st place solution in the US) for EY NextWave Data Science Competition 2019.

The aim of the competition is to investigate how data can help the next smart city thrive, and boost the mobility of the future.

![ey-cert][1]
[1]: /img/portfolio/ey-cert.png

#### Feature Engineering
##### Distance
First, I created some helper functions and converted the coordinates of the 4 egdes of the city centre to latitudes and longitudes. The other features I derived from these 2 are:
```
Distance features: haversine, manhattan, euclidean, and hausdorff
Lag features - previous location
Rotation
Distance to borders
```
![ey-distance][2]
[2]: /img/portfolio/ey-distance.png

##### Time
For each entry, I recorded the duration between entry and exit time and also the speed, which essentially is distance over time.

##### Grid size
I used created grid sizes of 10, 100, 200, 500, and 1000:
```
sizes = [10, 100, 200, 500, 1000]

# grid
def class_en_ex(df, sizes):
    for size in sizes:
        *jwd_class_entry, = map(lambda x,y:(int(x*size)%size)*size+(int(-y*size)%size), 
                            df["lat_entry"], 
                            df["lon_entry"])

        *jwd_class_exit, = map(lambda x,y:(int(x*size)%size)*size+(int(-y*size)%size), 
                            df["prev_lat_exit"], 
                            df["prev_lon_exit"])

        df["class_entry_" + str(size)] = jwd_class_entry
        df["class_exit_" + str(size)] = jwd_class_exit
```
##### Frequency
Frequency on latitudes and longitudes – to mimic congestion: 
```
def encode_FE(df, col, test):
    cv = df[col].value_counts()
    nm = col+'_FE'
    df[nm] = df[col].map(cv)
    test[nm] = test[col].map(cv)
    test[nm].fillna(0,inplace=True)
    if cv.max()<=255:
        df[nm] = df[nm].astype('uint8')
        test[nm] = test[nm].astype('uint8')
    else:
        df[nm] = df[nm].astype('uint16')
        test[nm] = test[nm].astype('uint16')        
    return  
```
Here, I demonstrate that having frequency encoding for the `class_entry_200` is beneficial for lightgbm because it is unable to capture the spikes in the orange bar. The only way for tree algorithms to capture such spikes is by performing a frequency encoding for the variables.

![ey-fe][3]
[3]: /img/portfolio/ey-fe.png


#### Statistical features
Since gradient boosting algorithms are unable to capture statistical features, I created the statistical features of the number of sessions, mean, standard deviation, median, max, and min for the top 30 variables `top_vars` that `lightgbm` deem are important.

#### Validation strategy
Below, I employed a `KFold` validation strategy with `k=5` since it aligns well with public leaderboard score. For each fold, the model will predict on the `val` and `test` set. The output for the test set is then averaged across all folds.

I also created a customized `f1_score` for `lightgbm` to ensure it maximizes `f1_score` instead of other binary classification metric.

Things are slightly interesting here. My initial strategy was to train on the training set which includes all sessions but the last session, validate on the last session from train_data and predict on the last session from test_data. My highest public leaderboard score for this approach was around 0.867.

I then switched to the `KFold` strategy of training and validating on only the final session from train_data and predict the last session from test_data. This approach boosted my score to 0.88 on the public leaderboard.

#### Model
My baseline was a `lightgbm` with `max_depth=5` and `learning_rate=0.05` before I tuned for hypermeters. I then tuned for hyperparameteres using `hyperopt`.

#### Feature Importance
Here I plot feature_importance ranked by `lightgbm`.

![ey-fe][4]
[4]: /img/portfolio/ey-fimport.png

#### Journey to Discovering Magic
I was stuck at 7th place for 3 weeks – public leaderboard score of 0.870. I then proceeded to discard all sessions other than the last session, which contradicts the saying “the more the merrier”. My final `train_data` had 134,063 entries and I leaped to the 2nd place. After tuning for hyperparameters using `hyperopt`, I was at the 1st place.