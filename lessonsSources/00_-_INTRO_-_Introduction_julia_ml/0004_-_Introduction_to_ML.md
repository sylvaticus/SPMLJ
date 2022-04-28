TODO. Please refer to the videos above or to the slides



# An introduction to Machine Learning

Now that we have introduced the programming language we'll use in this course, we can enter the second topic of the course, machine learning. But before that, let's just prepare a bit the work by clarifying the terminology.

## Terminology

Indeed, unless you are already well into the field, it is pretty likelly that by now you will have a certain confusion between the different terms that seem to be related, but that also seem to overlap.
What is the difference between _Artificial Intelligence_, _Machine Learning_, _Deep Learning_.. ? 
To add to the confusion, the chart below, that reports the frequency of various terms in Google Books across time, shows how the various terms become fashionable in different periods, sometimes becoming a buzzword, only to fall into disuse and then come back again...
![timeline ML terms](assets/imgs/mlTermsTrends.png)

Let's hence look at the various terms, with the help of a dictionary

### Artificial Intelligence 

Before we see the dictionary's definition for _artificial intelligence_, let's try to answer a question: would you say that a pocket calculator as the one depicted below uses Artificial Intelligence ? I bet your answer would be negative..

![timeline ML terms](assets/imgs/mlPocketCalculator.jpg)

Dictionary definitions for "artificial intelligence":
- _"The theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages."_ (Oxford)
- _"1 : a branch of computer science dealing with the simulation of intelligent behavior in computers"_
- _"2 : the capability of a machine to imitate intelligent human behavior"_ (Merriam-Webster)
- _"The study of how to produce machines that have some of the qualities that the human mind has, such as the ability to understand language, recognize pictures, solve problems, and learn"_ (Cambridge)

As you can read, the definition is given in comparative terms with human intelligence. But what is "intelligence" in the first instance? A definition reads as: "an intelligent agent has a goal, senses its environment and interacts with it, learns from the result of its actions and act toward achieving its goal, given perceptual limitations and finite computation" (Poole, David; Mackworth, Alan; Goebel, Randy, 1998. Computational Intelligence: A Logical Approach)

But the problem is that what is "normally requiring human intelligence" changes with time. Making aritmetic is surelly something deeply connected with human mind and the possibility that now a machine, a pocket calculator, could do it for us, should have been surelly considered a form of "artificial intelligent". However that sounds much more "obvious" for today's expectations, where we are used to much more. There is a lot of overlap with "progress" in the AI term.

It is still a term that remains pretty vague, so much that the definitions are provided in term of specific elements that are considered part of AI rather than by defining which are the common characteristics that something should have to be considered AI.

You can see in the chart that the term has been conied in the '50s, with diverse fortunes, and only recently has been "resurged" in specific fields.

I personally prefer not to use this term.

### Machine Learning

Let's look also for _machine learning_ into the dictionary definitions:

- _"The use and development of computer systems that are able to learn and adapt without following explicit instructions, by using algorithms and statistical models to analyse and draw inferences from patterns in data."_ (Oxford)
- _"The process by which a computer is able to improve its own performance (as in analyzing image files) by continuously incorporating new data into an existing statistical model"_ (Merriam-Webster)
- _"The process of computers changing the way they carry out tasks by learning from new data, without a human being needing to give instructions in the form of a program"_ (Cambridge)

A common concept in these definitions are that in ML we try to "learn" a system, its structure, some patterns or how the system relates its input to the outputs, from the data itself, without a predefined "model" of how this relation should be defined.

It follows that Machine learnings algorithms should be generic. They shouldn't be problem-specific, should not include domain-specific instructions, but rather should _learn_ the problem resolution from the data[^1].
[^1]: This is a remarcable and pretty desirable property to have: you will be surprised of how few algorithms can be applied to so many different contexts.

For example, for the digit recognition exercise that we will see in unit `NN`, we could have created a computer program that works with the following set of instructions: (1) identify the lines by checking for the variation in pixel darkness in a certain direction and when it is higher than a specific threeshold, consider it as a line: (2) assign the label "1" if you can find a vertical bar in the center of the image without any other lines in the middle (where I would have to define the tollerance for both "vertical" and "center"), and so on…
You can see that it is going to be complex. This is clearly not machine learning. Instead, machine learning should "learn" all these tasks from the data itself.

We can think to machine learning as a sub-field of artificial intelligence, and indeed it is hard to find success story of AI that are not ML.

Some definitions explicitly name statistical models. But while nothing forbid us to use statistical models as algorithms, statistical model aren't required. We don't need to make any probabilistic assumption concerning the distribution of our data, aside to respect that the data we train our model with has the same characteristics of the data we then we use it for.
The algorithm is tested, it is evaluated, not on specific assumptions, even if we use a statistical model, but by measuring its accuracy on data that has not been used in training the algorithm itself.


So, how does machine learning models hence work?
In its most simple way, a ML algorithm learns a relation between some x and the y, based on some data. This is the "training" step of the algorithm.
Once the model is trained, we use it to produce predictions based on other data, and the algorithm is evaluated on how far these predictions are from the real Y.
If we start to notice that the predictions are no longer good, we need to train again to the new data that we have.

While predictions aren't the only scope of machine learning, we'll see other kind of machine learning tasks in a couple of sections, it is undisputed that prediction is where machine learning shines, where there has been the greatest successes and the most practical applications.
Similarly, while predictions aren't surely the sole objective of science, many problems can be ultimately reframed with a prediciton approach.

But before we continue, a warning is due. Machine learning models, for the way they are trained, they may reflect and amplify the biases of the society rather than reduce them. If our training data is biased, this bias can make its way into our machine learning models. The output that we may obtain from these models, and the decisions we take based on these outputs, are not necessarily in line with the ethical views of the society.
So this is a very important societal issue, and I invite you to document by yourself about it, but it will not be covered in this course.


But why machine learning now ? Why this revival of this specific aspect of artificial intelligence? There are three pillars underpinning the "machine learnings boom" that we are currently experiencing.
The first one is the **algorithmic foundation*.  Algorithms have being improved, and we managed to find algorithms that are empirically very good in generalise a problem, in finding this relation between the X and the Y in general terms, and not restricted to the data that has been used for training them.
The second aspect is the **data availability**. Machine learning requires a lot of data. Often predictions become good only if the data the model are trained with is large enough. And the availability of data to train the algorithms has exploded with the digitalisation of the society. We are now able to collect a large amount of data, including personal data. Again, this opens many issues on the field of the privacy, that are not discussed in this course, but are important to consider.

Finally, we have the algorithms and we have the data, the last point is that only recently we obtained the **comutational power**, the means to process these huge amounts of data and run the algorithms over them. Think about the Graphical Processing Units for example, or the democratisation of cloud infrastructures.

So it is the combination of these three factors together that allowed for the rapid development of machine learning techniques in everyday aspects of the society.

### Deep Learning

To complete the definitions, the term "deep learning" refers to the usage of a specific kind of ML model, neural networks, when the model is used with multiple transformation layers between the inputs and the outputs. We'll see neural networks in detail in the `NN` unit, but let's here only notice that while it is difficult to think to succesful AI applications that are not ML, there are many ML algorithms in wide use that are not NN.


## A first example: estimating bike rental demand

As our first encounter with machine learning models, let us consider a task where need to estimate the demand of bike rental for the day (or the hours) in different rental locations, so that the management company can optimise the allocation of the resources across the varous spots.
As the input for our estimations we will consider various weather data (temperature, raining and windy conditions,...), the day of the week, the hour if we need an intra-day estimation, the season, some flags to tell us if the day is working, is a shool holiday or the following day is.
The object of our estimation is the number of bikes rented at that specific location, on that day (and eventually hour).

If we would go for a statistical model, we would write a _model_ like: `Bike rented = α +β₁*temp+β₂ * temp*wday+ …` where the modeller needs to make assumptions about the data and their relations, the variable to include, the covariance between variables..
The goodness of fit (significance) of the estimated parameters then would depend on the above assumptions.
After the model has been estimated, the modeller needs to provide the decision maker just the above equation and the `α, β₁, β₂, ...` parameters

If we go instead for a machine learning approach, we would use instead a "generic" regression algorithm (for example a decisions tree) and the goodness of the predicted output would be obtained by comparing the ML predictions with the true values for a set of records that has not being used to train the model (we'll see this important concept in detail in the `ML1` unit).
While ML predictions are likelly to be more accurate compared to those obtained by an out-of-sample interpolation of the statistical model, there is still a glitch.
Now we don't have any more a _compact_ representation of our model, so we can provide the company with just an equation and its parameters.
For the decision maker to be able to make predictions, we need now to provide to it the whole "trained" algorithm, so that it can be ran by the decision maker whenever it needs to make predicitons.

## Further examples

Digits recognition
will the algorithm recognise your hand-written digits ? (i.e. why captchas are a matter of the past) 
A regression task: the prediction of bike sharing demand
The task is to estimate the influence of several variables (like the weather, the season, the day of the week..) on the demand of shared bicycles, so that the authority in charge of the service can organise the service in the best way.
A classification task when labels are known - determining the country of origin of cars given the cars characteristics
In this exercise we have some car technical characteristics (mpg, horsepower,weight, model year...) and the country of origin and we would like to create a model such that the country of origin can be accurately predicted given the technical characteristics. As the information to predict is a multi-class one, this is a classification task. It is a challenging exercise due to the simultaneous presence of three factors: (1) presence of missing data; (2) unbalanced data - 254 out of 406 cars are US made; (3) small dataset.
A clustering task: the prediction of plant species from floral measures (the iris dataset)
The task is to estimate the species of a plant given some floral measurements. It use the classical "Iris" dataset. Note that in this example we are using clustering approaches, so we try to understand the "structure" of our data, without relying to actually knowing the true labels ("classes" or "factors"). However we have chosen a dataset for which the true labels are actually known, so to compare the accuracy of the algorithms we use, but these labels will not be used during the algorithms training.

## Type of Machine Learning areas

Supervised learning
obj: predict Y ("labels" or "target": continuous → regression or categorical → classification) based on X ("features")
given a certain number of (X,Y) couples provided to the algorithms
learn solution from examples
Unsupervised learning
exploit the "hidden" structure of some data (normally highly dimensional) without having any label provided
es. PCA, clustering
Reinforcement learning
learn the "best" action an agent can make toward a certain goal given unknown "rewards" of the actions
e.g. board games (chess, go), self-driving robots, control systems
often in a physic world
need training with many repeated "experiments" (including many failures!)
exploration vs exploitation trade off

## ML current research hot topics

ML algorithms differentiate for the predictive capacity, the computational efficiency (both memory and CPU), the necessity of input preprocessing (both simple data transformation and feature-engineering), the possibility to "reverse-engineer" the machinery and "understand" the relation between the X and the Y,...
requirement for lot of data, next step is transfer knowledge from a system (problem) to an other


## 