+++
image = "img/portfolio/grab.png"
showonlyimage = false
date = "2016-11-05T19:44:32+05:30"
title = "Grab AI for S.E.A. challenge - Safety"
draft = false
weight = 2
+++

Derive a model to detect dangerous driving trips
<!--more-->

#### Introduction
This post details an analysis and modeling for Grab AI for S.E.A. challenge - Safety.

#### Exploratory data analysis
I started my analysis with trip duration. It was found that the shortest trip time is ~2 minutes, which is rather unusual. This might be due to the rider cancelling the ride. Seeing the 10th percentile is ~6 minutes, I decide to remove trips taking less than 3 minutes and I will keep a flag for trips taking less than 5 minutes in the later part.

The longest trip time is 415,500 hours while the 99.9th percentile is 55 minutes. I will remove anything larger than 55 minutes.

#### Feature engineering
Before performing any feature engineering, the intuition I had on what distinguishes a dangerous trip from a safe trip are: aggresive turns, harsh breakings, and driving at high speed.

I took the first order derivative for the acceleration and gyroscope reading. I also calculated the roll and pitch as follows:
```
def roll(df):
    # roll  
    yz = df[['acceleration_y', 'acceleration_z']].values
    y, z = yz[:, 0], yz[:, 1]
       
    df['roll'] = np.arctan2(y, z)
    
def pitch(df):
    # pitch
    xyz = df[['acceleration_x', 'acceleration_y', 'acceleration_z']].values
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    df['pitch'] = np.arctan2(-x, np.sqrt(np.power(y, 2) + np.power(z, 2)))
```
I went on to take the aggregate/statistical features such as `min`, `max`, and `mean` for the acceleration and gyroscope readings. Besides, I counted the number of times the acceleration exceeded certain thresholds such as the following:
```
def count_above_2(x, thres=2):
    return (sum(x >= thres) + sum(x <= -thres))  
```

Knowing that impatient drivers are more likely to be dangerous drivers. A common charecteristic is the tendency to take alternative longer routes in lieu of congested main road or stopping for red lights.

This will result in them having travelled a longer distance in a shorter amount of time.

I intend to model this behaviour by assigning the label `too_far` to drivers who take roughly the same amount of travel time but has the `distance_sum` in the 90th percentile.

#### Modeling
After preprocessing and feature engineering, I moved on the modeling part. My best performing model is a `lightgbm` tuned using `hyperopt`.

#### Feature selection
For the first run, I selected all 167 features but after running `lightgbm` twice and narrowing the most important features into half each run, I am left with 41 features. The 41 features here manage to achieve a higher AUC-ROC score compared to 167 features.

Other features that are excluded from the statistics summary are Accuracy and Bearing. Accuracy refers to how accurate the GPS is in recording the device's position and Bearing is merely the direction the driver is heading. Those two do not correspond to the intuition on how they would be able to capture whether is a trip safe or dangerous.

#### Validation
For the validation part, I used `sklearn`'s `StratifiedKFold` with `k=5`. The final overall ROC-AUC score averaged over 5 folds is 0.731.

#### Observations
##### Leak features
max_second happens to be the most predictive variable to whether is a trip safe or dangerous. Removing this "feature" reduces the ROC-AUC from 0.720 to 0.680.

In my humble opinion, if Grab were to do real-time prediction, this feature has little value from the business or practical point of view. Nevertheless, I am drawing the conclusion that longer trips tend to be more dangerous based on the EDA shown on the previous notebook.

##### Top 3 importance features
Excluding max_second, the feature importance plot by lightgbm shows that the most predictive feature is horsepower_median, Speed_max, Speed_median. These make sense intuitively since driving at a higher speed indicates driving more dangerously.

#### Interpretation
To understand each feature from the best model, I ran `shap` on `lightgbm`.

![shap][1]

From the plot above, we can tell that summary statistics related to `second_max` has the highest impact on the `label`.

![tree][2]
Features here correspond to the intuition set up earlier. In each trip, we want to extract the maximum speed the driver achieved.

[1]: /img/portfolio/shap.png
[2]: /img/portfolio/tree.png