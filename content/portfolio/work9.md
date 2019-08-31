+++
draft = false
image = ""
showonlyimage = false
date = "2016-11-05T20:02:19+05:30"
title = "Earnings Call Sentiment Analysis"
weight = 9
+++

How positive can a CEO frame negative news during earnings call?
<!--more-->

#### Introduction
Every quarter, a public listed company will hold an earnings call to update shareholders on how has the company performed during the past quarter. During such events, CEOs have a higher tendency to frame negative news in a positive manner to assuage shareholders' concerns. This project strives to perform text mining and sentiment analysis on the earning calls transcripts.

#### Dataset
The dataset is scraped from The Motley Fool, fool.com. 

#### Approach
First, extract conversations by the CEO and then if the sentence contains adjectives related to a business or metrics, such as the follows, then we keep those sentences.
```
business_adj = [
    [{"LEMMA": {"IN": ["down", "up", "small", "big",
                       "high", "low", "strong", "weak",
                       "large", "bad", "solid"]}}],
]

metrics = [
    [{"LEMMA": {"IN": ["revenue", "cost", "margin", 
                       "grow", "profit", "sale",
                       "guidance"]}}], # sentence contains number?
]
```
Again, using `spaCy` to extract the patterns, here is an example from Infosys.
```
Matched on: strong
Infosys has delivered a strong quarter and I'm pleased with our overall performance as we continue to demonstrate our increasing relevance to clients. 
Adj describing: a strong quarter


Matched on: revenue
Our digital revenue growth was 41.9% and our digital revenue now accounts for 35.7% of our overall business. 


Matched on: large
The large deal DCV was the highest ever at $2.7 billion. 
Adj describing: The large deal


Matched on: margin
Our operating margin for Q1 was at 20.5%. 
```

#### Generating label
To generate the label, I looked at how have stock prices changed between the day before the earnings call was held and the day after the earnings call was held. Using `yfinance`, I was able to get the change in stock prices. If the stock price rises more than 5%, then the CEO is reporting positive news while falling more than 5% indicates bad news. Anything between -5% and +5% is ignored.

