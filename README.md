# Stand-Up Comedy and NLP

This natural language processing python project started from this tutorial https://www.youtube.com/watch?v=xvqsFTUsOmc. This project started because the instructor, Alice Zhao, saw a stand-up comedian for the first time and she really loved it. But she wanted to figure out what makes Ali Wong’s routine different than other comedians.

![](https://cdn-images-1.medium.com/max/2824/1*9YgP3LT62Ryl61h_005Zrg.png)

This project uses several NLP libraries including NLTK, TextBlob, gensim, as well as standard machine learning libraries like pandas and scikit-learn.

I’ll be using the the Jupyter Notebook IDE from the [Anaconda Distribution](https://www.anaconda.com/distribution/). I will post the entire Jupyter Notebook at the end of this blog as well as code snippets that you can follow along with as I go. There are a few python packages I will be using that are not included in Anaconda: wordcloud, textblob and gensim. They can be installed with the following commands:

    conda install -c conda-forge wordcloud

    conda install -c conda-forge textblob

    conda install -c conda-forge gensim

### In this end-to-end project I will:

1. Start with a question

1. Get and clean the data

1. Perform exploratory data analysis

1. Apply NLP techniques

1. Share insights

## 1. Start with a Question

The goal is to look at transcripts of various comedians and note their similarities and differences. Specifically, I’m trying to determine how Ali Wong’s comedy style is different than other comedians.

## 2. Data Gathering and Cleaning

**Here is the [full Jupyter Notebook](https://gist.github.com/nwams/da9290c4a21c1fddfc5cba9f82f8ba5a) with the entire code for this section.**

### Gathering the Transcript

I need to get the transcript from her Netflix special called “Baby Cobra”. But how/where do we get the transcript? Doing a Google search for her transcript shows that a site called [Scraps from the Loft](https://scrapsfromtheloft.com/) has full transcripts of several comedians’ routines as well as [Ali Wong’s](http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/).

When I inspect the elements using Chrome Developer Tools I can see that all of the transcript data is in the `<div class=“post-content”>`. Here’s a screenshot:

![](https://cdn-images-1.medium.com/max/2734/1*tu55ELv0S7kKGoSGGbvETQ.png)

Defining the scope of a machine learning project is a very important step. So in deciding how much data to gather, the scope will be limited to the following criteria:

1. Comedy specials within the past 5 years.

1. Use only comedy specials that have an [IMDB](https://www.imdb.com/?ref_=nv_home) rating of at least 7.5 stars and 2000+ votes.

1. Select the top 12 stand-up comics (this is where domain expertise is crucial. If you’re not sure you should do a sanity check with someone who knows the industry well).

In order to gather this data I will have to scrape the website. I will use the [Requests](https://3.python-requests.org/) package to get info from the website and the [Beautiful Soup](https://pypi.org/project/beautifulsoup4/) package to extract portions of the site.

Here is the code snippet for gathering the data:

https://medium.com/media/80cc3cc9934422b18aad9abf3d115587

Next, I created a dictionary named data where every key is a comedian and every value is the transcript.

I need the output in a clean and organized format that I can use for further analysis so I will be creating 1) a __Corpus__ and 2) a __Document-Term Matrix__. A Corpus is a collection of text and a Document-Term Matrix is just word counts in a matrix format.

### Creating a Corpus

In the code I go through a few steps to create a dictionary with the format {key: comedian, value: string} and then I convert it to a Pandas DataFrame. All of this can be found in the Jupyter Notebook.

Now on to the next step, cleaning the data! Why perform these cleaning steps? Because I want the computer to try to understand only the most important parts of the text.

Here are some common data cleaning steps that are usually done on all text:

* Make text lower case

* Remove punctuation

* Remove stop words

* Remove numerical values

* Remove common non-sensical text, like newline characters /n

* Tokenize the text

Below is a code snippet that shows how to clean the data.

https://medium.com/media/fa6ea06607d1f0eb2d8f2676ff9b968d

Here is my corpus:

![](https://cdn-images-1.medium.com/max/2818/1*xL-HvQ5hLYxXU1Vadwbxxw.png)

### Creating a Document-Term Matrix

Let’s walkthrough this concept using a line from [John Mulaney’s routine](http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/):
> “All right, Petunia. Wish me luck out there. You will die on August 7th, 2037.”

How can a computer read this text? I will need to 1) clean it 2) tokenize it and 3) put it into a matrix.

Remember in the section above I did some initial cleaning by removing punctuation, remove numbers and making all letters lowercase. After the first pass at cleaning it now looks like this:
> “all right petunia wish me luck out there you will die on august”

Now I will **tokenize** the text. Tokenization means splitting the text into smaller parts. Tokenization can be done as sentences, bi-grams or words. I will tokenize by ‘words’, so that every word will be its own item:

![](https://cdn-images-1.medium.com/max/2512/1*hF-7f8HjwV7r91JD7M8TwA.png)

Now I will remove **stop words** — words with very little meaning…like ‘a’, ‘the’, and ‘it’. After removing the stop words I’m left with:

![](https://cdn-images-1.medium.com/max/2496/1*Y-_vhYiRlCEBafe-yrnQ2A.png)

More simply put I’m left with just the following words:
> right petunia wish luck die august

This is called a **bag of words** model. Bag of words is just a group of text where the order doesn’t matter. It’s a simple and powerful way to represent text data!

Now using this I can create a **Document-Term Matrix** that counts the occurrence of words used by each comedian.

![](https://cdn-images-1.medium.com/max/2652/1*82RzsPQLQUdfaZIRxKeNDQ.png)

* The numbers are word counts

* Each row is a different document (or “transcript” for our case)

* Each column is a different term (or “word” for our case)

Here’s a code snippet below of the process of creating a document-term matrix. Let’s use scikit-learn’s CountVectorizer to tokenize and remove stop words.

https://medium.com/media/c6130662eed986e19a168e4ea5098851

Here’s a screenshot of the actual document term matrix. Notice that is has over 7,000 columns so it’s quite long. And it would be even longer if we had included bi-grams.

![](https://cdn-images-1.medium.com/max/2818/1*n5G0pav0EmZJMcULZ2WO5A.png)

If we wanted to further clean the text, after tokenization, there are more steps we could take. **Stemming / lemmatization** which is grouping words like “driving, drive, drives” as the same. We could do **Parts of speech tagging**, create **bi-grams** like turning “thank you” into one term and also **deal with typos**, etc.

## 3. Exploratory Data Analysis

**Here is the [full Jupyter Notebook](https://gist.github.com/nwams/d9219470fec8783e44c8062bd82ec99c) with the entire code for this section.**

Here’s the fun part! The goal of EDA is to summarize the main characteristics of the data, which will help us to determine if the trends make sense. It’s often best to do this visually. And it’s best to do EDA before applying machine learning algorithms.

I’m going to start by looking at the most common words used most by each comedian. I’ll also look at the size of their vocabulary because it would be interesting to see if some comedians have a higher vocabulary than others. And lastly I’ll explore the amount of profanity (this came after I saw the top words. You’ll notice that the comedians use a lot of cuss words).

Here we go!

### Top Words

Sort across the document-term matrix to find the top words. And visualize the data using [word clouds](http://amueller.github.io/word_cloud/). *As mentioned in the beginning of this blog the Word Cloud package is not included in Anaconda, so you’ll have to download it using this command: conda install -c conda-forge wordcloud.

I looked at the top 30 words said, per each comedian, along with a count of how many times it was said. I noticed that there were a lot of non-meaningful words such as “like”, “im”, and “know” amongst all comedians. The picture below of the top 15 words said by each comedian shows that a lot of stop words are included.

![Top 15 words said by each comedian.](https://cdn-images-1.medium.com/max/2580/1*5HAU71YBKftSmACOjMWW4Q.png)*Top 15 words said by each comedian.*

If these common words occur as top words by more than 50% of the comedians, then I’ll add those words to the list of stop words. Therefore I will be adding these to the list of stop words: [‘like’, ‘im’, ‘know’, ‘just’,‘dont’, ‘thats’, ‘right’, ‘people’, ‘youre’, ‘got’, ‘time’, ‘gonna’, ‘think’, ‘oh’, ‘yeah’, ‘said’].

After recreating a new document term matrix that includes the additional stop words, I made a word cloud to visualize the most common words for each comedian.

![](https://cdn-images-1.medium.com/max/2580/1*1L0dHHVz9e6nxIrux8Kelg.png)

Notice that Ali Wong says the S-word a lot, as well as ‘ok’ and ‘husband’. The instructor of this tutorial resonates with this because she says ‘ok’ a lot too and thinks her routine is funny because she also talks about her husband a lot too.

You’ll also notice that a lot of people say the F-word a lot, which we’ll also explore in a bit.

### Number of Words

I want to see how big of a vocabulary they have so I’ll count how many unique words they use. But another thing that I’ll look at is the number of words-per-minute based on how long their entire comedy special was.

Here’s a pandas dataframe that shows the unique_words and the words_per_minute.

![](https://cdn-images-1.medium.com/max/2580/1*bH44OIdigf9zeCGVrVc1AA.png)

I like to begin visualizing in the simplest ways possible first. So I’ll create a simple horizontal bar plot.

![](https://cdn-images-1.medium.com/max/2580/1*JZ0WfV6WNgohj72GAqonCw.png)

I can see that Ricky and Bill have a the highest vocabulary while Anthony has the lowest vocabulary. Additionally I can see that Joe Rogan talks the fastest while Anthony talks the slowest. But this visualization doesn’t yield any particularly interesting insights regarding Ali Wong, so let’s move on.

### Profanity

Remember in the word cloud above I noted that people say the F-word and the S-word a lot so I’m going to dive deeper into that by making a scatter plot that shows F-word and S-word usage.

![](https://cdn-images-1.medium.com/max/2446/1*8a4O8MEkrD9vy9pBuFsZjg.png)

Notice that Bill Burr, Joe Rogan, and Jim Jefferies use the F-word a lot. The instructor doesn’t like too much profanity so that might explain why she’s never heard of these guys. She likes clean humor so profanity could be a good indicator of the type of comedy she likes. Besides Ali Wong, her other two favorite comedians are John Mulaney and Mike Birbiglia. Notice how John and Mike are also very low on the use of profanity (Mike actually doesn’t use any S-words or F-words at all).

Remember that the goal of Exploratory Data Analysis is to take an initial look at the data to see if it makes sense. There’s always room for improvement like more intense clean up including more stop words, bi-grams etc. However perfection can serve to your disadvantage. The results, including profanity, are interesting and in general it makes sense. Since the science of machine learning is an iterative process, it’s best to get some reasonable results early on to determine whether your project is going in a successful direction or not. Delivering tangible results is key!

## 4. Apply NLP Techniques

## Sentiment Analysis

**Here’s the [full Jupyter Notebook](https://gist.github.com/nwams/e446596ce07b08386af2aedd3116f8a7) with the entire code for this section.**

Because order is important in sentiment analysis I will use the corpus, not the document-term matrix. I will use [TextBlob](https://textblob.readthedocs.io/en/dev/), a python library that provides rule-based sentiment scores.

For each comedian I will assign a **polarity** score that tells how positive or negative they are and a **subjectivity** score that tells how opinionated they are.

But before we jump into using the TextBlob module it’s important to understand what’s happening in the code, let’s take a look…

Under the hood: A linguist named Tom De Smedt created a lexicon ([en-sentiment.xml](https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/en/en-sentiment.xml)) where he manually assigned adjective words with polarity and subjectivity values.

![Subjectivity lexicon for English adjectives.](https://cdn-images-1.medium.com/max/2734/1*tHKaYFueVJFcAkXlCsdOkw.png)*Subjectivity lexicon for English adjectives.*

*From the image above, note that [WordNet](https://wordnet.princeton.edu/) is a large English dictionary created by Princeton.

I will output a polarity score which tells how positive/negative they are, and a subjectivity score which tells how opinionated they are.

From the [_text.py file in TextBlob’s github](https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/_text.py) there’s a section that defines the following: Words have a polarity (negative/positive of -1.0 to +1.0) and a subjectivity (objective/subjective of +0.0 to +1.0). The part-of-speech tags (pos) tags “NN”=noun and “JJ”=adjective. And the reliability specifies if an adjective was hand-tagged (1.0) or inferred (0.7). Negation words (e.g., “not”) reverse the polarity of the following word.

![](https://cdn-images-1.medium.com/max/2734/1*6c4tUUK0kCjnC9qusMq_6g.png)

I use the corpus during my sentiment analysis, not the document-term matrix, because order matters. For example, “great”=positive but “not great”=negative.

TextBlob handles negation by multiplying the polarity by -0.5. And handles modifier words by multiplying the subjectivity of the following word by 1.3. Since “great” has a subjectivity of 0.75, “very great” will have a subjectivity of 0.975 which is just 0.75*1.3.

Notice that the word “great” occurs in the [lexicon](https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/en/en-sentiment.xml) 4 times. In this situation textBlob will just take the average of the 4 scores. This is very basic and there’s nothing fancy or advanced happening here behind the scenes. So this is why it’s important for us to know what’s happening behind the scenes before we use a module.

Overall, TextBlob will find all of the words and phrases that it can assign a polarity and subjectivity score to, and it averages all of them together. So at the end of this task each comedian will be assigned **one polarity score** and **one subjectivity score**.

However be aware that this rules-based approach that TextBlob uses is not the most sophisticated, but it is a good starting point. There are also other statistical methods out there like Naive Bayes.

After creating a simple Sentiment Analysis scatter plot of the Polarity and the Subjectivity, I can see that Dave Chappelle’s routine is the most negative out of them all. And John Mulaney is most similar to Ali Wong, while Ricky Gervais isn’t too far away either.

![](https://cdn-images-1.medium.com/max/2000/1*hhK-L3FwzGeMqKTQ-mKxyg.png)

In addition to the overall sentiment, it would also be interesting to also look at the sentiment over time throughout each routine. So I’ll split each comedian’s routine into 10 chunks and assess their polarity pattern to see if I can draw additional insights.

![](https://cdn-images-1.medium.com/max/3028/1*WO3WFGm-JN350-qXIeo_Pw.png)

Ali Wong remained consistently positive throughout her routine. Louis C.K. and Mike Birbiglia are similar to her.

## Topic Modeling

**Here’s the [full Jupyter Notebook](https://gist.github.com/nwams/9a44423d8e787e678a7fe233fa051265) with the entire code for this section.**

Now I’d like to find themes across various comedy routines, and see which comedians tend to talk about which themes. In other words I want to see what topics are being said in this document. I will use the document-term matrix because order doesn’t matter, thus the bag-of-words model is a good starting point.

I will be using the [gensim](https://github.com/RaRe-Technologies/gensim) library, a Python toolkit built specifically for topic modeling, to apply a popular topic modeling technique called Latent Dirichlet Allocation (LDA). LDA is one of many topic modeling techniques. Also I will be using nltk for parts-of-speech tagging.

**What is LDA and how does it work?**

Latent means hidden and Dirichlet is a type of probability distribution, so I’m looking at the probability distribution in the text in order to find hidden topics. LDA is an unsupervised algorithm. You’ll find that LDA is really useful when you have really large documents and have no idea what they’re about.

Here are two simple but important definitions. Every **document** consists of a mix of **topics**. And every **topic** consists of a mix of **words**.

![](https://cdn-images-1.medium.com/max/3028/1*Fz7E-amdBJE6owJKl3VHrQ.png)

The goal is for LDA to learn what all of the topics in the document and what are all of the words in each topic.

Every topic is a probability distribution of words, something like this:

* Topic A: 40% banana, 30% kale, 10% breakfast…

* Topic B: 30% kitten, 20% puppy, 10% frog, 5% cute…

Here’s a visual summary of what LDA is about:

![](https://cdn-images-1.medium.com/max/3028/1*ZgQPCBuxf5rtx1fPF5FH-Q.png)

Here’s how LDA works:

1. You choose the number of topics you think are in your corpus (Example: K=2)

1. LDA randomly, and temporarily assigns each word in each document to one of the 2 topics (The word ‘banana’ in Document #1 is randomly assigned to Topic B the animal-like topic).

1. LDA will go through every word & its assigned topic and it will update the topic assignments. So let’s say ‘banana’ is assigned to the Animal topic. It will decide whether it should re-assign it to the Food topic by first checking how often the Animal topic occurs in the document, then secondly it will check how often the word ‘banana’ occurs in the Animal topic. Both of those probabilities are low so it will re-assign ‘banana’ to the other Food topic. Here’s the math behind it:
`Proportion of words in document d that are currently assigned to topic`.
`t = p(topic t | document d)`.
And `Proportion of assignments to topic t over all documents that come from this word`.
`w=p(word w | topic t)`.
Multiply those two proportions and assign ***w*** a new topic based on that probability.
`p(topic t | document d)*p(word w | topic t)`.

1. You have to go through multiple iterations of previous step. Go through a few dozen iterations yourself and eventually the topics should start making sense. If the topics don’t make sense then more data-cleaning is needed.

The Gensim package will do steps 2 and 3 for you. It is up to you to set the number of topics you want (step 1), and how many iterations you want to go through (step 4).

Gensim will output the **top words in each topic**. It’s your job as a human to interpret the results to figure out what the topics are. If not, try altering the parameters, either: Terms in the document-term matrix, number of topics, iterations, etc. **Stop when you’re able to come up with topics that make sense.**

### Topic Modeling with All of the Text

In my first attempt I set the number of topics to 2, and the number of times the algorithm will pass over the whole matrix to 10. I then trained the LDA Model using **all** of the words from my term-document matrix. I’ll start by setting the number of topics to 2, assess, and then increment if needed.

These are the words output from the LDA model when number of topics is 2:

![](https://cdn-images-1.medium.com/max/3028/1*fcqNFKRjlTYrb5Zt-LQ6sw.png)

It’s not making sense yet and there’s overlap in the words. Here’s the output when number of topics is 3:

![num_topics=3](https://cdn-images-1.medium.com/max/3028/1*V1afx7KqjDjOd_PHg5DOew.png)*num_topics=3*

It’s not quite making sense yet. Let’s increment again by setting the number of topics to 4 to see if it improves:

![](https://cdn-images-1.medium.com/max/3028/1*tDKRQbTF2Ad4q_reyUfnxQ.png)

These topics aren’t too meaningful.

### Topic Modeling with Nouns

One popular technique is to only look at terms that are from one part of speech. So I’ll try modifying the bag of words to include only **nouns**. The results below are from trying 2, 3 and 4 topics respectively.

![](https://cdn-images-1.medium.com/max/2440/1*lpNMPB55sMWF7wE9BxYE7w.png)

Again, there isn’t much improvement and there’s still overlap.

### Topic Modeling with Nouns and Adjectives

So in this next attempt I will add **adjectives**. My LDA model will now assess nouns and adjectives. The results are below.

![](https://cdn-images-1.medium.com/max/2430/1*q9jda0KUWPnVNrbVbc-uBQ.png)

As you can see there still isn’t much improvement. So I experimented with tuning the hyper-parameters, increasing the number of passes from 10 to 100. And setting *alpha* and *eta* values. For *alpha* I experimented with really low values as well as symmetric and auto. For *eta* I experimented with low values in an attempt to get less “overlap” between topics.

*Alpha* and *eta* are hyper-parameters in Gensims LdaModel that I can tune.

A high *alpha* value means that every document is likely to contain a mixture of most of the topics, not just any single topic specifically. While a low *alpha* value means that a document is more likely to be represented by just a few of the topics.

A high *eta* value means that each topic is likely to contain a mixture of most of the words, not just any specific word. While a low *eta* value means a topic is more likely to contain a mixture of just a few of the words.

Simply put a high *alpha* will make documents appear more similar to each other. And a high *eta* will make topics appear more similar to each other.

Unfortunately tuning the hyper-parameters did not yield any meaningful topics. I also tried including **verbs** and retraining the model with nouns, adjectives and verbs but that didn’t help it either.

### **Why my data isn’t ideal for Topic Modeling**

The model assumes that every chunk of text that we feed into it contains words that are somehow related. So starting with the right corpus is crucial. However, comedy specials are inherently dynamic in nature with no fixed topics in most streams. Since the subject matter is constantly switching throughout a comedian’s routine there usually isn’t one centralized topic. Whereas, in contrast, if we trained our LDA model with Wikipedia articles, each article (document) is already highly contextualized as it usually talks about a single topic, which is a good thing. It’s also good to note that the number of documents per topic is also important. LDA is extremely dependent on the words used in a corpus and **how frequently they show up**.

## Wrapping Up

In this project I did Text Pre-Processing, Exploratory Data Analysis, Sentiment Analysis, and Topic Modeling.

![](https://cdn-images-1.medium.com/max/3004/1*uKRgMp8vzRUrG5ylmNyPUw.png)

## **NLP Libraries**

Here are some popular NLTK libraries to be aware of and some brief details.

* NLTK — **This is the library that everyone starts with**. It has a lot of text-pre-processing capabilities like tokenization, stemming, parts-of-speech tagging, etc.

* TextBlob — This was built on top of NLTK, is **easy to use**, and includes some additional functionality like sentiment analysis and spell check.

* Gensim — This library was built **specifically for topic modeling** and include multiple techniques including LDA and LSI. It can also calculate document similarity.

* SpaCy — This is the newest of the bunch and is known for its **fast performance** since it was written in [Cython](https://cython.org/). It can do a lot of things that NLTK can do.

## 5. Summary of Insights

Remember the original question the instructor wanted to answer was “What makes Ali Wong’s comedy routine stand out?”

* Ali talks about her husband a lot, and the instructor also talks about her husband a lot during her lectures.

* In terms of profanity, Ali had the highest s-word to f-word ratio. The instructor doesn’t mind the s-word but does not like hearing the f-word at all.

* Ali Wong tends to be more positive and less opinionated, which is similar to the instructors personality as well.

Based on these findings, who are some other comedians the instructor might like?

* Comedians who don’t say the f-word that often: Mike Birbiglia (no curse words at all) and John Mulaney.

* Comedians with a similar sentiment pattern: Louis C.K. and Mike Birbiglia.

[Mike Birbiglia](https://www.imdb.com/title/tt2937390/?ref_=fn_al_tt_1) is a comedian that she’d probably like as well!
