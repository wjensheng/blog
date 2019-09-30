+++
draft = false
image = "img/portfolio/nbalogo.jpg"
showonlyimage = false
date = "2016-11-05T19:50:47+05:30"
title = "NBA Hackathon 2019"
weight = 6
+++

Predict engagement of Instagram posts by @nba
<!--more-->

#### Introduction
This post presents the solution for NBA Hackathon business analytics track. 

All participants were tasked with predicting the "engagement" of Instagram posts by @nba. It was first noted by @nba that "engagement" is not the number of likes of a post.

#### Adversarial validation
Given a training set and test set, I wanted to decide if I should perform a random split or time-based split for my validation set. I set out to do adversarial validation on both the training and test set.

The idea is to make the model predict whether is an entry a training or test set by assigning a label of 1 to the training set and 0 to the test set.

If the AUC-ROC is around 0.5, it means the model cannot tell whether is does the entry belong to the training or test set. In that case, I am able to perform a random split for my valdation.

After running a `lightgbm` model for 5 folds, the AUC-ROC for the above dataset is indeed, around 0.5.

#### Preprocessing
I extracted some time features based on the datetime of a post:
```
Hour, minute, second
Day of the week
Month
Quarter
Year
Number of posts on the day
Number of posts 2 days before (since playoffs have a one-day gap)
```

For the description of a post, I processed the text and applied `FastText` embeddings to them. Surprisingly, after some cleaning, I was able to find embeddings for 91% of the text. I then converted the 300-dimension embeddigns to 300 columns of features.

Here are some rather unusual contraction mappings that I fixed manually:
```
contraction_mapping = {"tonight's": "tonight is",
                       "nbaplayoffs": "NBA playoffs",
                       "thisiswhyweplay": "this is why we play",
                       "kingjames": "lebron james",
                       "nbaallstar": "NBA all-star",
                       "nbaonabc": "NBA on ABC",
                       "nbaontnt": "NBA on TNT",
                       "nbapreseason": "NBA preseason",
                       "nbatv": "NBA TV",
                       "nbafinals": "NBA finals",
                       "nbabreakdown": "NBA breakdown",
                       "nbaonespn": "NBA on ESPN",
                       "russwest44": "russell westbrook",
                       "kyrieirving": "kyrie irving",
                       "nbakicks": "NBA kicks",
                       "nbaday": "NBA birthday",
                       "nbasummer": "NBA summer",
                       "tripledoublealert": "triple double alert",
                       "phantomcam": "phantom camera"}
```

#### Text features
I also extracted the following feature for each document - Description.
```
Number of #
Number of @
Number of punctuations
Number of uppercase letters
Number of tokens
```

#### Exploratory data analysis
Number of posts with time appears to be the most interesting plot.
![post-time][1]

#### Modeling
The metric used by @nba was mean absolute percentage error. I then ran an `xgboost` model optimizing for this metric:
```
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def xgb_mape(y_true, dtrain):
    labels = dtrain.get_label()
    return 'mape', np.mean(np.abs((y_true - labels) / y_true)) * 100
```
Using `StratifiedKFold` with `k=5`, the average MAPE across all 5 folds was 3.46%.

#### Interpretation
![shap][2]
Apparently, the type of the most (photo of video) and the number of @ signs appears to affect the "engagement" level fairly significantly!

[1]: /img/portfolio/nba-post.png
[2]: /img/portfolio/nba-shap.png