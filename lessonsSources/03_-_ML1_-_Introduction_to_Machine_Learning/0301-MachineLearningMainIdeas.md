# 0301 - Machine Learning main ideas

We start our journey on Machine Learning from the perceptron algorithm. This is a linear classifier that historically has been a bit a pioneer in ML algorithms. We start from it due to its simplicity, and it will be the only algorithm that we will study in detail.

## A linear classifier for the binary classification problem

So, which is our problem? We are in the realm of supervised machine learning as we defined in the kick-off lesson, where the objective is to make some predictions over a component of the dataset (an unknown characteristic, or some future event...) that we call "label" based on other known characteristics of the data (that we call "features"), and we can "learn" this relation by "supervising" the algorithm, that is providing the algorithm with data from which we know both the features and the associated labels (across this course I'll often use "X" as a shortcut for the former, and "Y" for the latter).
More specifically we are in the realm of classification, where the labels are given a finite set of possible values, and we start from binary labels, where the values can only be -1 or +1 (it is easy then to use this binary classifier as the base for a multiclass classifier, e.g. using a so-called _one-vs-all_ strategy).
The features are numerical, and possibly highly multi-dimensional.

The Perceptron algorithm is a method to find the parameters of a linear classifier that minimise classification errors. These classifiers will be nothing else than a hyperplane (the generalisation of the concept of "plane" on multiple dimensions) that divides our space into two parts, with the hyperplane itself forming the boundaries between the two parts, and we will use this hyperplane to discriminate all points on one side of it v.s. all points on the other side.
For example, the following figure shows a hyperplane in 2D (i.e. a line) used to classify some points:

![A linear classifier in 2D](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/linearClassifier.png)

All points in the direction of the arrow are "positive", all points in the opposite direction are negative.
From the figure we can deduce a few things in that example:
- the current classifier (as drawn) is NOT classifying all points correctly, ie it makes "errors"
- the points are however linearly separable
- it would be enough to "rotate" a bit the classifier clockward to get the classifier making no more errors

The perceptron algorithm is indeed an online algorithm (that is, that is updated as data is processed..) to "rotate" the classifier until it minimises the classification errors. For simplicity, we will work with classifiers passing through the origin, but we don't lose in generality as we can always think of the constant term as another dimension where the feature data is all filled with ones.

Our first step is to "decide" how to define the classifier, and how to check if it classifies a record correctly or not.
Concerning the former, we define the boundary given by the parameters $\theta$ as the set of points $x$ for which $x \cdot \theta = 0$ (the inner product is equal to zero).
It results that all points $x$ for which $ x \cdot \theta > 0$ are classified positively, and similarly all points $x$ for which $ x \cdot \theta < 0$ are classified negatively.
For a given record, we can now check if this classifier matches the label of the record (-1 or +1). In analytical terms, we have a classification error of the (linear) classifier $\theta$ for the record $n$ when  $y^n * (\theta \cdot x^n)) \leq 0$.
The overall error rate of the classifier will then be the sum of the errors for each record divided by the number of records $N$:

$\epsilon_n(\theta) = \sum_{n=1}^N \frac{ \mathbb{1} [ y^n * (\theta \cdot x^n ) \leq 0 ] }{n}$ where the one at the numerator is an indicator function. And note that we consider a point exactly on the boundary still as a classification error.


## Supervised learning

Before we can turn to the problem of actually finding a linear classifier that agrees with the data to the extent possible, that is looking at the specific perceptron algorithm, we need to develop a method to generally deal with "learning algorithms" in the context of supervised learning, where - like here - we are given a set of both the features (the "X" matrix) and the associated labels (the "Y" vector - or sometimes matrix).

Let's then define as the algorithm's **parameters** those parameters that are learned by an algorithm by processing the (X,y) data that we provide to it to learn its relations.
Often an algorithm has also so-called **hyperparameters**. These are additional parameters that remain constant during the learning (also called _training_) step, while still affecting the performances of the algorithm. For example, the number of neurons in a neural network layer, or the number of individual decision trees in the random forest algorithm. Or, for many algorithms (including perceptron) how _long_ should the learning step continue (this can take different forms depending on the algorithm).

Hyperparameters play a fundamental role in the trade-off between specialisation and generalisation.
The objective of the training is not indeed to learn a relation between the given X and the given Y, but rather to learn from the provided data the _general_ relationship between X and Y for all the populations from which X and Y have been sampled. And hyperparameters should be "set" to the levels that maximise this objective, not the minimisation of errors in the data used to train the algorithm. If we use too many neurons, if we train too much,.. we would learn the _specific_ relation between X and Y in the training data. However, this reflects the specific data provided and not the general population. In statistical terminology, we would _overfit_ our model, or in other words, generate too much _variance_ in the trained parameter of our model, that would depend too much on the specific training data (i.e. different training data would lead to very different learned parameters).
On the opposite, if our model is too simple or receives too little training, we will have too much _bias_ and not learn the relationship sufficiently. Techniques that allow an algorithm to better generalise, at the expense of better performances over the training data, are called "regularisation".

How can we choose the hyperparameters that minimise the bias-variance trade-off ? We put no assumptions on the data except that they all come from the same population.
And the idea is to use the data itself to "evaluate" the generality of our model.
We randomly split our dataset into three subsets:
- The **training set** is the one used to actually "train" the algorithm to learn the relation between the given X and the given y, provided a certain set of hyperparameters, that is to find the parameters than minimise the error made by the algorithm
- the **validation set** is used to evaluate the results of our trained algorithm on data that has not been used by the algorithm to train the parameters, that is to find the hyperparameters that allow for the best generalisation
- finally the **test set** is used to judge the overall performances of the algorithm when it is used with the "best" hyperparameter (we can't use the validation set for this, as the hyperparameters are "fitted" based on it).


![Train, validation and test set](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/trainingValidationTestSet.png)

In practice, we have various ways to look for the "best hyperparameters".. grid search over the hyperparameters space, random search, gradient-based methods... In all cases, we run the algorithm under the training set, and we evaluate it under the validation set until we find the "best" hyperparameter set. At this point, with the "best" hyperparameters we train one last time the algorithm using the training set and we evaluate it under the test set.

### K-folds cross validation

Note that training and validation sets don't need to be the same sample across the whole process.
Indeed a common technique is the so-called **K-folds cross-validation**:

![5-folds CrossValidation](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/5FoldsCrossValidation.png)

Here we first randomly divide our whole dataset in a train/validation set and in the test set.
For each possible hyperparameter set, we randomly partition the train/validation test in K sets. We use K-1 of them for training and the remaining one for computing the out-of-sample score of the model. We do that (keeping the same hyperparameters and the same partition) for all the different K subsets and we average the performances of the model with that given hyperparameters.
We then select the "best" hyperparameters and we run the final training on the train/validation set and evaluation on the test set.

## The perceptron algorithm

We can now start our analysis of the Perceptron algorithm.

- We start with the parameters of the hyperplane all zeros $\theta = 0$
- We check if, with this parameter, the classifier makes an error
- If so, we progressively update the classifier using as the update function  $ere are better way\theta^{n} = \theta^{n - 1} + y^n * x^i$.

As we start with $\theta^0 = \begin{bmatrix}0\\0\\...\end{bmatrix}$, the first attempt will always lead to an error and to a first "update" that will be $\theta^{1} = \begin{bmatrix}0\\0\\...\end{bmatrix} + y^1 * \begin{bmatrix}x^1_{d1}\\x^1_{d2}\\...\end{bmatrix}$.

Let's make an exaple in 2 dimensions, with two points $x^1 = \begin{bmatrix}2\\4\end{bmatrix}$ and $x^2 = \begin{bmatrix}-6\\1\end{bmatrix}$, both with negative labels.

After being confronted with the first point, the classifier $\theta^0$ undergos its first update to become $\theta^1 = \begin{bmatrix}0\\0\end{bmatrix} + -1 * \begin{bmatrix}2\\4\end{bmatrix} = \begin{bmatrix}-2\\-4\end{bmatrix}$.
Let's continue with the second point, $x^{(2)} = \begin{bmatrix}-6\\1\end{bmatrix}$. Does $\theta^1$ make an error in classifying $x^{2}$ ? We have:  $y^{(2)} * \theta^1 \cdot x^{(2)} = -1 * \begin{bmatrix}-2\\-4\end{bmatrix} \begin{bmatrix}-6\\1\end{bmatrix} = -8$, so yes, we have another classification error.
We hence run a second update to obtain  $\theta^2 = \begin{bmatrix}-2\\-4\end{bmatrix} + -1 * \begin{bmatrix}-6\\1\end{bmatrix} = \begin{bmatrix}4\\-5\end{bmatrix}$.
I let you see geometrically that this classifier correctly classify the two points:


![Perceptron example over 2 points](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/perceptron2PointsExample.png)

More in general, we run the perceptron algorithm over the whole training set, starting from the 0 parameter vector and then going over all the training examples.
And if the n-th example is a mistake, then we perform that update that we just discussed.

So we moved the parameters in the right direction, based on an individual update.
However, since the different training examples might update the parameters in different directions, the later updates might also "undo" some of the earlier updates, and some of the earlier examples would no longer be correctly classified.
In other words, there may be cases where the perceptron algorithm needs to go over the training set multiple times before a separable solution is found (I let you try as exercise what would happen if the second point is $[+1,-2]$ instead of $[-6,+1]$, still with negative label).

So we have to go through the training set here multiple times. In the jargon of machine learning, we call _epoch_ each time an algorithm goes through the whole training set, either in order or selecting at random.
On each record, we look at whether the current classify makes a mistake and eventually perform a simple update.

* function perceptron $\displaystyle \left(\big \{ (x^{(n)}, y^{(n)}), n=1,...,N\big \} , epochs \right)$:
  * initialize $\theta =0$ (vector);
  * for $t=1,...,epochs$ do
    * for $n=1,...,N$ do
      * if $y^{(n)}(\theta \cdot x^{(n)}) \leq 0$ then
        * update $\theta = \theta + y^{(n)}x^{(n)}$
  * return $\theta$

So the perceptron algorithm takes two parameters: the training set of data (pairs feature vectors => label) and the epochs parameter that tells how many times going over the training set.

For a sufficiently large number of epochs, if there exists a linear classifier through the origin that correctly classifies the training samples (i.e. the training set is linearly separable), this simple algorithm actually will find a solution to that problem.
Typically, there are many solutions, but the algorithm will find one. And note that the one found is not, in general, the "optimal" one, where the points are "best" separated, just one where the points _are_ separated.

## Support Vector Machines: "better" linear classifiers

While the Perceptron algorithm finds _one_ possible classifier, it is clear that this may not be the "best" one. See the following figure:

![Different linear classifiers over the same dataset](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/differentLinearClassifiers.png)

Linear classifiers do generalise relatively well and the `epochs` parameter could be used as a form of regularisation. Still, we could end up with a perceptron classifier like in figure (a), very dependent on noisy data and that doesn't generalise well.
Support Vector Machines (SVM, which we will not develop further in this course except in this small discussion) are linear classifiers that try to maximise the boundary with the classified data. So here we enter the realm of "optimisation", usually achieved employing a gradient-based approach that we'll see when discussing neural networks: it is no longer indifferent one classifier that separates the data from another, but SVM (should) retrieve _the_ classifier that is more "far away" from the two datasets, on both the directions (figure (b)).
The second important characteristic of SVM is that this optimisation can be "adjusted" to consider not only the points that lie closest to the decision surface (the "support vectors", from which the name..) but rather to give importance to the points that are farther away from the boundary. This adjustment takes the form of a regularisation parameter that can try to optimize with the cross-validation technique above. So, in the figure example, we intuitively see that the third classifier ( figure (c) ) would be the best for our dataset, it will better match with the nature of our data, even if it would make some classification mistakes, and we can steer an SVM algorithm toward the one depicted on figure (c) by increasing its regularisation parameter.

## Using linear classifiers for non-linear classification

Often the relationship between the X and the Y is not linear in nature, and even employing the best linear classifier would result in significant classification errors (same for regressions).
The "good news" is that we can easily engineer our data performing non-linear transformation over it and still using a linear classifier.

For example in the left diagram in the figure below, the three points are not separable in one dimension:  

![Inseparable points becoming separable in higher dimensions](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/03_-_ML1_-_Introduction_to_Machine_Learning/imgs/separablePointsInHigherDimensions.png)

However, we can "engineer" our dataset by creating a new dimension that is the square of our original dimension (diagram on the right): the points are now clearly separable by a linear classifier !


The "bad news" is that this engineering is not "learned" by the algorithm, but it is something we still do on a case-by-case, using our expertise of the specific problem on hand. 
This is the main difference between linear algorithms used together with feature transformation and more "modern" non-linear algorithms where the non-linearity is learned from the data itself, like in trees-based approaches and neural networks. 









