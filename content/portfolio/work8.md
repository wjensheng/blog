+++
showonlyimage = false
draft = false
image = "img/portfolio/datahack.jpg"
date = "2016-11-05T19:59:22+05:30"
title = "2019 MDataHack"
weight = 8
+++

Predict ideal position for hockey players based on body movements
<!--more-->

#### Introduction
I participated in my very first Datathon organized by Michigan Data Science Team. In a team of 3, we were given 3 datasets and had to choose one to work on. After some discussion, we decided to work on the Catapult wearable device dataset.

#### Dataset
The dataset dated from August 2016 to September 2018 and consisted of GPS, Inertial, Player Load collected from Catapult Wearable Devices. A total of data points for 31 anonymized unique players from Michigan Hockey Team was available to us.

#### Position descriptions
To start, there are a few positions in a hockey team. Each position has a different function. As starters:

Forwards: Very fast, quick reflexes, stay on offensive end

Midfielders: In great shape, run the most, help on offense and defense

Backs: Stay in the defensive zone, side to side movements, aggressive

Goalkeepers: Stay in goal circle, very little movement

#### Hypothesis
"Total Player Load" gives us the best prediction of a player’s positions. We believed that certain positions require different levels of work. Players who are capable of putting in more work perform better in certain positions.

#### Data cleaning
We started with 21,168 rows of data and went on to clean the dataset by removing some outliers and dropping some useless features.
```
# drop useless features
df.drop(['Period Number', 'Date_obj', 'Start Time', 'End Time', 'Day Name', 'Month Name'], axis=1, inplace=True)

# relabel target
df.loc[df['Position Name'] == 'Forward/Mid', 'Position Name'] = 'Mid/Forward'

# filter out bad high speed distance
df = df[(df["High Speed Distance"] < 1000)]
```

#### Exploratory data analysis

![md-pos][1]
[1]: /img/portfolio/md-pos.png

![md-hm][2]
[2]: /img/portfolio/md-hm.png

![md-dist][3]
[3]: /img/portfolio/md-dist.png


#### Feature engineering
We attempted to create some features based on positions.
```
df['forward_features'] = df['High Speed Distance'] + 
						   df['Maximum Velocity'] + 
						   df['Velocity Band 5 Total Distance']
```

#### Preprocessing
We normalized all the statistics based on players.
```
players = df["ID"].unique()
for player in players:
	ids = df[df["ID"] == player].index
	scaler = MinMaxScaler()
	df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])

LabelEncoder()
```

#### Modeling
We used `XGBoost` with a total of 27 features. By using different `n_estimators` and `learning_rate`, the best model achieved an accuracy of 92.7% with 4,504 of training data.

```
n_estimators=500, learning_rate=0.2
n_estimators=100, learning_rate=0.05
n_estimators=200, learning_rate=0.1
```

#### Mystery feature
Before performing any feature engineering, the feature importance from `XGBoost` is as follows:
![md-fe1][4]
[4]: /img/portfolio/md-fe1.png

But after adding a mystery feature, the new feature importance chart is as follows:
![md-fe2][5]
[5]: /img/portfolio/md-fe2.png

Here is the mystery factor for players of different positions:
![md-fe3][6]
[6]: /img/portfolio/md-fe3.png

#### Concluding remarks
This might be an interesting finding for hockey coaches to decide upon what position should a player be playing. In the end of the day, we do not want Christiano Ronaldo to be a goalkeeper, right?
![md-ron][7]
[7]: /img/portfolio/md-ron.png