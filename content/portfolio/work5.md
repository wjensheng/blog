+++
image = "img/portfolio/cv.jpg"
showonlyimage = false
draft = false
date = "2016-11-05T19:53:42+05:30"
title = "Career Village"
weight = 5
+++

Develop a method to recommend relevant questions to the professionals who are most likely to answer them
<!--more-->

![cv-prize][3]
[3]: /img/portfolio/cv-prize.png


#### Introduction
The U.S. has almost 500 students for every guidance counselor. Underserved youth lack the network to find their career role models, making CareerVillage.org the only option for millions of young people in America and around the globe with nowhere else to turn.

My goal is to develop a method to recommend relevant questions to the professionals who are most likely to answer them.

First, Jared, C.E.O. of CareerVillage.org shared the following feedback from professionals:

```
"These question don't really feel relevant to me."
"I waited around but never got matched to anything."
"I used to get the emails, but then they stopped."
"This is way too much email. I get something like almost every day!"
"If the questions were more relevant, I'd be willing to get emailed more often. But if they are not relevant, then I don't want much email at all."
"The emails aren't that insightful, and don't show me enough info about the questions without having to click in"
```

All these revolve around two main themes, relevance of questions and frequency of receiving questions. After looking at CareerVillage's current recommender system, I realize it mainly matches professionals to questions based on tags. Hence, I plan to:

Introduce a recommender system that takes "questions" and "answers profile" into account using 4 different models as I will show how professionals have answered questions that not necessarily have tags they follow.

Introduce a new notification setting. This allows professional to have daily or 3-day (or more) notification setting. The system is able to limit the number of emails sent to professional per day and also keep professionals who have not been receiving questions engaged with the platform.

I will also provide some future recommendations:
```
Option to modify profile
Three-day email notification
Reducing spam by students
"Shifting" questions
Handling cold start by tags
Keeping track of questions that are unanswered
```

#### Exploratory data analysis
After some EDA, I noticed that most questions and answers are added on Wednesday, with weekends having the lowest activity.

In the next section, I define slow response time as more than 7 days taken for the first answer to be added; fast response as 3 days of less; moderate as between slow and fast.

![wordcount][1]
Questions that have a longer response time tend to be longer. We can perhaps solve this by targeting a larger audience in the RecSys.

I explore the common and rare words in the slow response category. Surprisingly, the common words from slow are in fact common - I can tell that there will be relevant professionals who are able to answer those questions. On the other hand, the uncommon words from slow seem to be more specific to region as words such as "australia", "europe", and "germany" appear.

#### Model
Four models will be deployed to generate the RecSys, namely TF-IDF FastText, GloVe, and Universal Sentence Encoder. The reason is all four approaches are able to capture various subtleties in a question. For instance, TF-IDF captures similarity between bigrams while FastText embedding allows similar semantics to be captured.

#### Evaluation
##### Evaluation Approach #1
The first approach to test our RecSys is by entering a query to get similar questions. Essentially, our model can be evaluated based on a query (entered manually) and check how similar is it to the questions recommended by the RecSys. I intend to make a note that CareerVillage can use this approach to recommend questions that have been answered in the past without necessarily recommending the query to the professionals.

Consider the following example:
```
query = 'i am intereted in data science and machine learning. do you have any tips for me?'
```
and the result:
```
(array([ 49, 831, 751, 847, 518, 488, 675,   9, 552, 592]),
 array([ 49,  46, 531, 488, 518, 416, 909, 351, 284, 398]),
 array([127, 651, 416,  46, 436, 297, 971,  49, 798, 199]),
 array([ 49,  46,  14,   4, 740, 284, 531, 894, 609, 260]))
```
Notice how all four models recommended different questions but there is still some overlap of indices of questions recommended.

A weight is then assigned to each question based on Borda count. `questions_id` that appears first has a weight of 1 and a decrement of 0.1 is applied to the subsequent `questions_id`, i.e., the second question has a weight of 0.9, third 0.8 and so on.

```
Recommedation by TF-IDF ...
1
What is the difference between data science and machine learning?
I've been looking into data science careers, and I know that it is closely related with machine lear


2
How did you complete a science college degree as an adult?
I am pursuing a science degree after taking a few years away from school.  Knowing science and math 


3
How to make your mind more creative?
#creativity #learning #mental-exercise #creative #learning 
-----
Recommedation by FastText ...
1
What is the difference between data science and machine learning?
I've been looking into data science careers, and I know that it is closely related with machine lear


2
Best resources for developing in Information Security?
Such as resources for learning specific programming/scripting languages (Python, Perl, bash), sites 


3
Which program is best to start learning coding?
I want to learn to code. Should I start with python, Java, etc. ?
#coding 
#computer-science 
-----
Recommedation by GloVe ...
1
What do you have to have to be a EMT
#savealife
#EMSislife


2
What is copywriting as a career, and what is your typical workflow?
#copy #writing #write #marketing #advertising #adobe #outreach #graphic #design


3
What is the best coding language to know to become a computer programer?
#computer #computer-engineering #computer-programming #coding 
-----
Recommedation by Universal Sentence Encoder ...
1
What is the difference between data science and machine learning?
I've been looking into data science careers, and I know that it is closely related with machine lear


2
Best resources for developing in Information Security?
Such as resources for learning specific programming/scripting languages (Python, Perl, bash), sites 


3
Can i be a biologist in Indianan. How much will i make. What else will i have to know.
i am looking for a job that allows me to be a biologist that study all types of DNA and animals, pla
-----
```

##### Evaluation Approach #2
Clearly, the RecSys can then recommend professionals based on professionals who have answered the similar questions shown above. But I think the RecSys should be able to further improve its recommendation if each professional has its own `profile`. My motivation of coming with this approach is `questions_body` or `questions_title` might not reflect the essence of the questions but answers can. I attempt to construct a `profile` for professionals based on what `vocab` best describes them. Given a query, the RecSys can recommend relevant professionals based on their historical answers, again building upon the four models.

##### Evaluation Approach #3
When recommending questions to professionals, the final RecSys accounts for
```
the similarity between query and other questions;
the similarity between query and professionals' profiles (containing vocab and score) based on their historical answers.
```
Now, Borda count is applied to the professionals_id based on the order they were suggested to construct our final dictionary p_counts, which consists of professionals_id and score_weight. In order words, the RecSys includes professionals who have answered similar questions in the past and professionals who have profiles (historical answers) similar to the incoming query.

The result now:
```
Recommending rank 1 professional
Headline: Student at California State University-Los Angeles
Top 3 highest scores words:
        vocab     score
0  algorithms  0.487712
1        data  0.478284
2    includes  0.390998


Recommending rank 2 professional
Headline: Business Intelligence Manager
Top 3 highest scores words:
              vocab     score
0  machine learning  0.422301
1      data science  0.414403
2           machine  0.400947


Recommending rank 3 professional
Headline: Data Scientist at Airbnb
Top 3 highest scores words:
              vocab     score
0          requires  0.277327
1  machine learning  0.261464
2           problem  0.256825
```

##### Evaluation Approach #4
It is now time to answer the question: will the RecSys be performing better than CareerVillage's current RecSys? Before diving into the approach, let's explore questions that have been recommended by CareerVillage with some EDA. The professionals dataset has a total of 28,152 professionals registered but EDA shows that < 1,000 remain active.

I would like to extend my gratitude to RodH for coming up with the following approach of filtering emails received and questions answered by emails.

Before I select a professional to conduct a further analysis, I intend to look for some professionals who have good track record, i.e., those who answered more than 20 questions and has an average response time of less than 5 days.

Consider the following person:
```
Professional's headline: Commercial Sales at Dell

CareerVillage suggested 99 questions
Professional answered 71 questions
0 question(s) suggested were answered
```

Interestingly, the professional answered 71 questions, none of which has been suggested by CareerVillage. Why is this the case?

Examples of questions suggested by CV:
```
1. On average how much does someone with a Computer Science degree make right out of school?
2. information technology or computer science which of these hard in math?
3. why do you like technology and try programing a computer?
4. I'm interested in a career in information technology. Are there jobs in aviation you suggest I look into?
5. What subjects are required for computer programming and computer engineering  and what jobs will you get with those careers?
```
Examples of questions answered by TEST_P_ID:
```
1. Should I live on campus or should I save money by living in an apartment around the town or city of the university?
2. In what ways is technology utilized by doctors?
3. what are some of the requirements to becoming a radiology tech
4. what courses are affiliated with a major in marketing?
5. What is the best college to attend to play football.
```

As seen above, the questions suggested were in fact, relevant to the professional's expertise, which revolves around computer science and IT. But the questions answered by `TEST_P_ID` are somewhat different from his or her expertise, including questions about doctor, radiology, and football.

#### Future recommendations
##### Recommendation #1: Option to modify profile
Although I am building a RecSys that helps to recommend relevant questions to professionals, I am thinking of having professionals to improve the RecSys. This approach is especially useful when a professional intends to answer questions he or she is interested in. For instance, the IT professional above could be interested in and have some knowledge about "marketing" or have experiences in the role in the past.

I suggest adding the option of allowing professionals to "update" their profile. After displaying the `vocab_score`, professionals are given the option to weigh certain word that they think describe themselves better or words they are more interested in answering heavier. This recommendation could also help resolve the issue we had in Evaluation Approach #4 when professionals have been answering questions not quite related to their expertise.

##### Recommendation #2: Three-day email notification
My second recommendation is to add a three-day email notification setting. I will illustrate this experiment over a three-day period. First, I randomly assign an email notification setting of daily and every 3-day (can be extended to weekly or monthly) to all the professionals in `p_ids`.

On day 1, I look at the questions posted and recommend 5 professionals to each question. If the professionals happen to be in the group of `three_day_noti` we recommend the question to the professionals and mark them in `dont_ask`, which is equivalent to removing them from our suggested pool. If the professionals happen to be in group of `daily_noti` we keep them.

Repeat the same process for day 2 and day 3.

###### Recommendation #3: Reducing spam by students
Should we handle spam by students?

Some students, 3 of them shown here, have been asking more than 5 questions in a single day. As long as the number of relevant professionals is large amount, the RecSys will be able to handle such situation by assigning questions to other than the top 5 professionals. Otherwise, CareerVillage should setup a system to reduce spam by students by limiting number of questions asked.

###### Recommendation #4: "Shifting" questions
We would like to tackle the problems CareerVillage is facing, mainly the feedback from professionals. Some are as follows:

"I waited around but never got matched to anything."
"This is way too much email. I get something like almost every day!"
I think churn might originate in two extreme cases: "superstar" professionals (a.k.a. professionals who received too many emails) and "marginal" professionals (a.k.a. professionals who seldom receive emails) I will approach both professionals differently. I suggest capping the number of questions in the former case and recommending a question to the "marginal" professionals who have not received any question in the past 2 days.

In the later case, I will convert the questions the "marginal" professional has answered in the past into what I have been calling query to look for similar questions in the questions_d3. This is illustrated below.

##### Recommendation #5: Handling cold start by tags
To handle the cold start problem, we suggest our new professional to select some of his or her favorite tags. (It is unlikely we can build a profile immediately for the professional by asking he or her to enter answers to a bunch of relevant and irrelevant questions.) Hence, I figured that TF-IDF is the best candidate to parse the tags selected by our new professional.


#### Metadata
##### Location
After some EDA, we come to realize that the pool of students is certainly diverse. This led to concerns about how some questions are very much pertained to a certain location. For instance, Priyanka asked about the life in the military in India. How can we handle this?

We relabel every state in the US to USA since professionals_location and `students_location` are referring to the state rather than the country. This will help us better handle questions raised from different countries. A problem I encounter is my inability to capture all US states, especially Greater Los Angeles Area since I could not split it properly. A suggestion is perhaps to standardize selection options for countries and states.

##### Answer scores
This could be an approach to determine quality of answers and professional's reputation.

###### Timeliness
As mentioned, timeliness is also an important metric to CareerVillage.

Interestingly, there are 36 instances of targeted professionals receiving the emails of questions they have already answered. It would be great if CareerVillage can handle such cases of prevent sending targeted professionals who have answered the question.

For the sake of analysis and providing a recommendation, I will convert the aforementioned professionals' resp_time to 0.

Subsequently, I will apply a negative exponential function to the resp_time. Logically, professionals who have a shorter response time should have a higher score.

 propose a formula to calculate time_score, which is
 ```
timescore=((2.5×exp(−0.04×numberofquestionsansweredsofar)+0.5)×responsescore
```
This formula accounts for the notion of "beginner's bonus" in the score. Hence, I designed a rewarding mechanism that gives high bonuses to professionals when they are freshcomers but lesser as they answer more questions. Eventually, this boils down to their timeliness. This allows for comparison between professionals who answer 100 questions and take 10 days to answer each question and professionals who answer 10 questions and take 1 day to answer each question - the more timely professional.

The number 2.5 and -0.04 are hyperparameters that can be tuned. They correspond to what the initial bonus is and how quickly the bonus decreases as the user answers more questions. The offset 0.5 is used because the mutlipler should eventually approach the value 0.5 as the number of questions answered by a professional approaches infinity.

##### Previous interactions
Knowing the previous interactions will not be very beneficial in recommending questions given there are many 1-time and 2-time interactions.

#### Meta-RecSys
In the final section, I demonstrate a meta RecSys that takes into account metadata while recommending questions. In sum, four components (with weights assigned) will be taken into account while recommending questions.

Ranking: 2.5
Ranking is an important factor as it accounts for similarity between questions and how likely professionals are able to answer.

Timeliness: 2
Timeliness is the next important metric since we want quetions to be answered ASAP.

Heart_score: 0.75
Hearts score serves as a proxy for quality of answers.

Location: 5
As seen in the EDA section, most questions that are unanswered or have slow responses are country-specific, hence, a much higher weight is assigned to the location component.

#### Final Meta-RecSys
![cv][2]


[1]: /img/portfolio/cv-word.png
[2]: /img/portfolio/cv-diagram.png