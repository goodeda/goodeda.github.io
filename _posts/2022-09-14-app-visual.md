---
layout: post
categories: English
title: App analysis with review information on google play
tags: Visualization, POWER-BI
toc: true
pin: true
date: 2022-09-14 23:13 +0300
---
## Introduction
Each international student coming to finland has to consider opening a bank account for strong authentication which is a necessity in Finland. The access to identify yourself online greatly facilitate the administration. Besides, you need a bank account to make consumptions, transfer, get salaries or make a loan. There are three major banks that people use everyday.
I used python api and scraped reviews of those three applications. By visualising the data, I make it easier for people to have a quick look at the reputation of these three banks.

## Report

<iframe title="app_visual" width="600" height="486" src="https://app.powerbi.com/view?r=eyJrIjoiYjU5YWU2ODQtNjY5NS00OTFiLWFmN2EtMjljM2I3YmU4ZDQ5IiwidCI6IjcwZTNmODZhLWRjZTYtNDBhZS05MzYxLWY3NWY0MmE1ODZhMyIsImMiOjh9&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>

## Analysis/Interpretation
The data is automatically collected with google-play-scrape API and it's as of the beginning of September so might not be the latest values shown on google play store. I only analyze the results I got and won't make any personal judgement on the apps.

### Overall score
Nordea has the most highest average (4.16) score among the three banks. The rating of Danske bank is the lowest, 2.61/5.0.  

### Relation between Review length and score
Looking at the plot on upper right side, there is an interesting trend of review length and score. If the score goes higher, the review tends to be shorter. Otherwise, people complain more and the ratings drop. This is particulary true in Nordea. We can see a nosedive in score over the time of June, 2018. The plot actually makes sense because if customers are satisfyied with the app, they usually just comment with "hyva" "helppo" or "good". However, if the app broke down or some bugs happened, users are more likely to complain a lot about the problems. 

Something must happened in 2018 as all apps witnessed a dramastic drop on the ratings. It could be caused by version update or some regulations.  

### Review composition
From the doughnut chart, we can see Finnish users are dominant in three apps. English is the second largest part and swedish follows as the third.   

### Customer service reply
I used an area plot to show the reply rates of different banks. Although Danske is rated as the lowest, its customer service is making the best to reply each comments. The two almost fully overlapping areas indicates the reply rate is very high.     

In terms of Nordea, because of a larger number of comments, comparing with the other two, it replies not so frequently. The large gap exists also because most comments are positive (rating is high) and there is probably no need to make a response. OP bank's performance is the average level.   

### App time
Based on the collected data, we also know the operating time of Nordea and OP banks is longer, from 2014 to now, while the danske bank entered into application market since 2017. To some extent, I also believe the duration of the app influences its performance because if a app exists longer, it has more time to adjust and update versions to improve its usability.  

## Conclusion  
There are actually more information can be explored. For example, what did customer complain about in reviews? Does frequent iteration of versions affect the app reputation?   
Sentiment analysis is feasible but it could be challenging to do so as there are just a few models in dealing with Finnish texts.    

I feel quite proud of this mini-report because I do it from scratch, from data collection, data wrangling to final visualization. There must be many places can be improved and I wish to develop my visualization skills more in the future projects or work.

