+++
date = "2016-11-05T19:41:01+05:30"
title = "Christie's Art Auction"
draft = false
image = "img/portfolio/mundi.jpg"
showonlyimage = false
weight = 1
+++

Exploratory data analysis on artwork sold on Christie's, an auction house
<!--more-->

#### Introduction
This post presents an analysis of Christie's Post War & Contemporary Art. 

Contrast the following examples.

![dollar1][1]
![dollar2][2]

Two similar artwork but prices differ by approximately a 100 times. How fascinating! This certainly piqued my interest in exploring what determines price of an artwork.

To do so, I went on Christie's Post War & Contemporary Art page to scrape some data. I took a look at the `robot.txt` and it did not prohibit me from scraping the page.

After a detailed analysis, it was found that the most expensive paintings ever sold was Salvador Mundi by Leonardo da Vinci. This painting is also know as the male version of Mona Lisa. It was sold for 450 mil USD. The most popular artist, measured in terms of number of paintings sold, was Andy Warhol, who painted the famous Campbell's Soup Can.

New York hosted the most auctions, followed by London. 

The year 2009 had the lowest sales record ever since 2006 while 2015 was the best year.

Oil is the most popular traditional paint medium.

Artwork executed between year 1890 to 1914 fetched the highest average realized price.

Amedeo Modigliani had the highest average realized price for his artworks. According to Wikipedia, he is best known for portraits and nudes in a modern style characterized by elongation of faces or necks.

The painting that has the biggest difference between its upper estimate and realized price was Mark Rothko's Orange, Red, Yellow. Its higher estimate was 45 mil USD but was realized at 86.9 mil USD. 

There were around 70 items that have appeared more than once on the auction market. An example was Crio+Cristo+Critico. It was first sold at 19,100 EUR in 2007 and resold for 12,800 EUR in 2008.

#### Technical Details
##### Data collection
I wrote a script to crawl Christie's based on `selenium` and `BeautifulSoup`. Here is all information I collected and its corresponding missing values.
```
title              0
auction_id         0
auction_loc        0
sale_tot           0
parent             0
date               0
item_id            0
item_name       1245
item_past          0
img_url         3191
item_des           0
item_prov      10861
est_price        667
real_price       669
```

After collecting the urls of the images, I proceeded to download the actual images. Here are the libraries I have used:
```
import multiprocessing
from io import BytesIO
from urllib import request
from PIL import Image
```

##### Data cleaning
For all prices, I converted `str` into `int` and removed some commas to ensure consistency.

Since auctions are held at various locations throughout the world, they are prized in the domestic currency. To convert all currencies to USD, I went to https://www.bis.org/statistics/xrusd.htm to download exchange rates back to the year 2006. I only accounted for the monthly exchange rate. Very interestingly, I learned how to slice dataframe columns with the following function `np.r_`. Here is a snippet of the code:
```
# fx.columns[3502:3663] # '2006-01'
# fx.columns[3662] # '2019-05'

# fx = fx.iloc[:,np.r_[3:5,3502:3663]] # only countries and monthly
```
Here is a mapping function to convert non-USD currency to USD
```
def find_fx(curr, ym):
    if curr == 'USD': return 1
    return fx.loc[curr, ym] # fx is the exchange rate table  
```

##### Anomaly detection
It was noticed that some items do not have an artist. I suspect this is due to some different formatting issues the crawler did not manage to handle.

Instead, these items' artist is, in fact, the item_name. Items alike were tricky to look for among the 50,000 odd items. But a trend was observed that such items have all uppercase titles and starts with the determiner "A". Hence, `spaCy` was used to look for such items by checking for determiner.

After retrieving the indices of such items, I swapped the `item_name` with `artist`.

In total, there are around 3,484 "mislabeled" items.

##### Looking for similar images
It occurred naturally to ask whether is there any item that has been auctioned more than once, i.e., exchanged hands?

To do so, my approach is to look for similar items based on image hashes. This was done with the `imagehash`.

I was able to make use of the `pandarallel`'s `parallel_apply` function to speed up the process twice the original needed time.

##### Extracting text
Most items have descriptions that contain the medium used and dimensions. To extract the text, I applied `spaCy`'s `Matcher` to look for patterns. 

With the following pattern, I was able to extract the dimensions of an artwork.

```
dim_pat = [
    [{"POS": {"IN": ["NOUN", "NUM"]}},
     {"POS": "SYM"}, 
     {"POS": "NUM"},
     {'POS': "SYM", "OP": "*"},
     {"POS": "NUM", "OP": "*"},
     {"LOWER": "c"},
     {"LOWER": "m"}
    ],
    
    [{"LOWER": "c"},
     {"LOWER": "m"},
     {"POS": "NUM"},
     {'POS': "SYM", "OP": "*"},
     {"POS": "NUM", "OP": "*"}     
    ]   
]
```

Most importantly, I learned how to use `spaCy`'s pipeline and `python`'s iterator to speed up the process. Here is a snippet:
```
def pipe_matcher(matcher, docs):
    for doc in docs:
        matches = matcher(doc)
        if len(matches) > 0:
            match_id, start, end = matches[-1]    
            span = doc[start:end]  
            yield span.text    
            
        else:
            yield ''   

mt = pipe_matcher(matcher, docs)

df['dims'] = tqdm(list(mt))
```
It only took 5 minutes to 48,343 entries!

In the end, I successfully extracted 44.48% traditional art media from and 82.31% of dimension the descriptions for all the entries!

##### Further details
Please check https://nbviewer.jupyter.org/github/wjensheng/post_war_art/blob/master/01-christies_eda.ipynb out to see the above findings!

[1]: /img/portfolio/dollar1.png
[2]: /img/portfolio/dollar2.png