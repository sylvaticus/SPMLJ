# 0401 - Neural Networks - theory

While powerful, neural networks are really composed of very simple units that are akin to linear classification or regression methods.

We'll first describe them, and we'll later learn how to train a neural network from the data.
Concerning the practical implementation in Julia, we'll not implement a complete neural network but rather only certain parts, as we will mostly deal with using them and apply to different kind of datasets.

## Motivations and types

We already saw the Perceptron algorithm as a linear classifier, but we also noted how we can transform the original feature vector $\mathbf{x}$ to feature representation of $\phi(\mathbf{x}$ where the transformation could well be non-linear, so to still use linear classifiers (we saw for classification but exactly the same is true for making regressions).
The "problem" is that this feature transformation is not learn from the data but it is applied a priori, before using the actual machine learning (linear) algorithm.
With neural networks instead, the feature transformation is endogenous to the learning (training) step.

We will see three kinds of neural networks:
- **feed-forward neural networks**, the simplest one where the inputs flow through a set of "layers" to reach an output. 
- **convolutional neural networks**, where one of more of the layers is a "convolutional" layer. These are used mainly for image classification
- **recurrend neurla netorks (RNN)**, where the input arrive not only at the beginning but also at each layer. RNNs are used to learn sequences of data


## Feed-forward neural networks

### Description 

In **deep forward neural networks**, neural network units are arranged in **layers**, from the _input layer_, where each unit holds the input coordinate, through various _hidden layer_ transformations, until the actual _output_ of the model:

![Neural network scheme](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/feedforwardNNChart.png)


More in detail, considering a single _dense_ neuron (in the sense that is connected with _all_ the previous layer's neurons or with the input layer), we have the following figure:

![Single neuron](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/singleNeuron.png)

where:
- $x$ is a due dimensional input, with $x_1$ and $x_2$ being the two dimensions of our input data (they could equivalently be the outputs of a previous 2 neurons layers)
- $w$ are the _weigth_ that are applied to $x$ plus a constant term ($w_0$). **These are the parameter we will want to learn with our algorithm**. $f$ is a function (often non-linear) that is applied to $w_0 + x_1w_1 + x_2w_2$ to define the output of the neuron

The output of the neuron can be the output of our neural network or it can be the input of a further layer.

Let's specific a bit of terminology concerning Naural Networks:

- The individual computation units of a layer are known as **nodes** or **neurons**.
- **Width_l** (_of the layer_) is the number of units in that specific layer $l$
- **Depth** (_of the architecture_) is number of layers of the overall transformation before arriving to the final output
- The **weights** are denoted with $w$ and are what we want the algorithm to learn.
- Each node's **aggregated input** is given by $z = \sum_{i=1}^d x_i w_i + w_0$ (or, in vector form, $z = \mathbf{x} \cdot \mathbf{w} + w_0$, with $z \in \mathbb{R}, \mathbf{x} \in \mathbb{R}^d, \mathbf{w} \in \mathbb{R}^d$) and $d$ is the width of the previous layer (or the input layer)
- The output of the neuron is the result of a non-linear transformation of the aggregated input called **activation function** $f = f(z)$
- A **neural network unit** is a primitive neural network that consists of only the “input layer", and an output layer with only one output.
- **hidden layers** are the layers that are not dealing directly with the input nor the output layers 
- **Deep neural networks** are neural network with at least one hidden layer

While the weights will be learned, the width of each layer, the number of layers and the activation funcions are all elements that can be tuned as hyperparameters of the model, altought there are some more or less formal "rules":

- the input layer is equal to the dimensions of the input data
- the output layer is equal to the dimensions of the output data. This is tipically a scalar in a regression, but it is equal to the number of categories in a multi-class classification, where each "output dimension" will be the probability associated to that given class
- the number of hidden layers reflects our judgment on how many "levels" we should decompose our input to arrive to the concept expressed in the label $y$ (we'll see this point dealing with image classification and convolutional networks). Often 1-2 hidden layers are enought for classical regression/classification. 
- the number of neurons should give some "flexibility" to the architecture without exploding too much the number of parameters. An heuristic is to use a number of neurons ~20% higher than the input dimension. This is often fine-tuned using cross-validation as it risks to lead to overfitting
- the activation function of the layers except the last one is choosen between a bunch of activation functions, nowadays it is almost always used a simple _Rectified Linear Unit_ function, aka `relu`, defined as `relu(x) = max(0,x)`. The relu function has the advantage to add non-linearity to the transformation while remaining fast to compute (including the derivative) and avoiding the problem of vanishing or exploding the gradient (we'll see this aspect when dealing with the actual algorithm to obtain the weigths). An other coomon choice is `tanh()`, the hyperbolic tangent function, that maps with a "S" shape the real line to the interval [-1,1].
- the activation function of the last layer depends on the nature of the labels we want the network to compute: if these are positive scalars we can use also here the `relu`, if we are doing a binary classification we can use the `sigmoid` function defined as `sigmoid(x) = 1/(1+exp(-x))` whose output is in the range [0,1] and which we can interpret as the probability of the class that we encode as `1`. If we are doing a multi-class classification we can use the `softmax` function whose output is a PMF of probabilities for each class. It is defined as $softmax(x,k) = \frac{e^{x_k}}{\sum_{j=1}^{K} e^{x_j}}$ where $x$ is the input vector, $K$ its length and k the specific position of the class for which we want to retieve its "probability".

Let's now make an example of a single layer, single neuron with a 2D input `x=[2,4]`, weights `w=[2,1]`, `w₀ = 2` and activation function `f(x)=sin(x)`.

In such case the output of our network is `sin(2+2*2+4*1)`, i.e. -0.54. Note that with many neurons and many layers this becomes essentially (computationally) a problem of matrix multiplications, but matrix multiplication is easily parallelisable by the underliying BLAS/LAPACK libraries or, even better, by using GPU or TPU hardware, and running neural networks (and computing their gradients) is at the core of the demand for GPU computation.  

Let's now assume that the true label that we know beeing associated with our $x$ is `y=-0.6`.

Out (basic) network did pretty well, but still did an _error_: -0.6 is not -0.54. The last element of a neural network is indeed to define an error metric (the **loss function**) between the output computed by the neural network and the true label. Commonly used loss functions are the squared l-2 norm (i.e. $\epsilon = \mid \mid \hat y - y \mid\mid ^2$) for regression tasks and cross-entropy (i.e. $\epsilon = - \sum_d p_d  * log(\hat p_d)$) for classification jobs.

Before moving to the next section, where we will study how to put everything together and learn how to train the neural network in order to reduce this error, let's first observe that neural networks are powerful tools that can work on many sort of data, but they require however it to be encoded in a numerical form, as the computation is strictly numerical. If I have a categorical variable for example, I'll need to encode it expanding it to a set of dimensions where each dimension represent a single class and I encode with a indicator function if my record is that particular class or not. This is the most simple form of encoding and takes the name of _one hot encoding_:

![One-hot encoding](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/onehotencoding.png)

Note in the figure that using all the three columns leads to linearly dependancy, and while, yes, we could save a bit of resources by using only two columns instead of three, this is not a fundamental problem like it would be in a statistical analysis. 

### Training of a feed-forward neural network

#### Gradient and learning rate

We now need a way to _learn_ the parameters from the data, and a common way is to try to reduce the contribution of the individual parameter to the error made by the network. We need first to find the link between the individual parameter and the output of the loss function, that is how the error change when we change the parameter. But this is nothing else than the derivate of the loss function with respect to the parameter. In our simple one-neuron example above we have the parameters directly appearing in the loss function. Considering the squared error as lost we have $\epsilon = (y - sin(w_0 + w_1 x_1 + w_2 x_2))^2$. If we are interested in the $w_1$ parameter we can compute the derivate of the error with respect to it using the chain rule as $\frac{\partial\epsilon}{\partial w_1} = 2*(y - sin(w_0 + w_1 x_1 + w_2 x_2)) * - cos(w_0 + w_1 x_1 + w_2 x_2) * x_1$.

Numerically, we have: $\frac{\partial\epsilon}{\partial w_1} = 2(-0.6-sin(2+4+4)) * -cos(2+4+4) * 2 = -0.188$ If I increase $w_1$ of 0.01, I should have my error moving of $-0.01*0.188 = -0.0018$. Indeed, if I compute the original error I have $\epsilon^{t=0} = 0.00313$, but after having moved $w_1$ to 2.01, the output of the neural network chain would now be $\hat y^{t=1} = 0.561$ and its error lowered to $\epsilon^{t=1} =  0.00154$. The difference is $0.00159$, slighly lower in absolute terms than what we computed with the derivate, $0.0018$. The reason, of course, is that the derivate is a concept at the margin, when the step tends to zero.

We should note a few things:
- the derivate depends on the level of $w_1$. "zero" is almost always a bad starting point (as the derivatives of previous layers will be zero). Various initialisation strategies are used, but all involve sampling randomly the initial parameters under a certain range
- the derivate depends also on the data on which we are currently operating, $x$ and $y$. If we consider different data we will obtain different derivates
- while extending our simple example to even a few more layers would seems to make the above exercise extremely complex, it remains just an application of the chain rule, and we can compute the derivatives efficiently by making firstly a _forward passage_, computing (and storing) the values of the chain at each layer, and then making a _backward passage_ by computing (and storing) the derivatives with the chain rule backwars from the last to the first layer
- the fact that the computation of the derivates for a layer includes the _multiplication_ of the derivates for all the other layers means that if these are very small (big) the overall derivate may vanish (explode). This is a serious problem with neural networks and one of the main reasons why simple activation functions as the `relu` are preferred.

The derivate(s) of the error with respect to the various parameters is called the _gradient_.

If the gradient with respect to a parameter is negative, like in our example, it means that if we _increase_ slighly the parameter we will have a lower error. At the opposite, if it is positive, if we _reduce_ the parameter slighly we should find a lower error.

A gradient-descent based algorithm can hence be used to look iteractivelly for the minimum error by moving _against_ the gradient with a certain step.
The most basic algorithm is then $w_i^t = w_i^{t-1} - \frac{\partial\epsilon^{t-1}}{\partial w_1^{t-1}} * \lambda$ where $\lambda$ is the step that you are willing to make against the gradient, also known as _learning rate_.
Note in the example above that if instead of moving the parameter $w_1$ of $0.01$ we would have increased it of $0.1$, we would have increased the error to $0.00997$ instead of reducing it. This highlights the problem to use a good learning rate (see next Figure): a too small learning rate will make the learning slow and with the risk to get trappen in a local minima instead of a global one. Conversly, a too large learning rate will risk the algorithm to diverge.

![Learning rate effect](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/learningRateEffect.png)

So, the learning rate is also a hyper-parameter to calibrate, altougth some modern gradient descent variations, like the ADAptive Moment estimation (ADAM) optimisation algorithm, tend to self_tune themselves and we rarelly need to calibrate the default values.

#### Batches and Stochastic gradient descent

We already note that the computation of the gradient depends on the data levels. We can then move betweed two extremes: on one extreme we compute the gradient as the average of those computed on all datapoints and we apply the optimisation algorithm to this average. On the other extreme we sample randomly record by record and at each record we move the parameter.
The compromise is to partition the data in a set of _batches_, compute the average gradient of each batch and at each time update the parameter with the optimisation algorithm.
The "one record at the time" is the slowest approach, but also is very sensitive to the presence of outliers. The "take the average of all the data" approach is faster in running a sertain epoch, but it takes longer to converge (i.e. it requires more epoches, the number of times we pass througth the whole training data). It also require more memory, as we need to store the gradients with respect to all records.
So, the "batch" approach is a good compromise, and we normally set the batch number to a multiplier of the number of threads in the machine performing the training, as this step is often paralleilised, and it represents a further parameter that we can cross-validate.
When we sample the records (individually or in batch) before running the optimisation algorithm we speak of _stochastic gradient descent_.

## Convolutional neural networks

### Motivations 

Despite typically classified separatelly, convolutional neural networks are essentially feed-forward neural networks with the only specification that one or more layers is a convolutional layer.
These layers are very good in recognising patterns within data with many dimensions, like spatial data or images, where each pixel can be tought a dimension of a singol record.
In both cases we could use "normal" feed-forward neural networks, but convolutional layers offer two big advantages:

1. _They require much less parameters_.
Convolutional neurons are made of small _filters_ (or _kernels_) (typically 3x3 or 5x5) where the same weigth convolves across the image. Converselly if we would like to process with a dense layer a mid-size resolution image of $1000 \times 1000$ pixels, each layer would need a weight matrix connecting all these 10^6 pixel in input with 10^6 pixel in output, i.e. 10^12 weights
2. _They can extend globaly what they learn "locally"_
If we train a feed-forward network to recognise cars, and it happens that our training photos have the cars all in the bottom half, then the network would not recognise a car in the top half, as these would activate different neurons. Instead convolutional layers can learn indipendently on where the individual features apply.

### Description

In these networks the layer $l$ is obtained by operating over the image at layer $l-1$ a small **filter** (or **kernel**) that is slid across the image with a step of 1 (typically) or more pixels at the time. The step is called **stride**, while the whole process of sliding the filter throughout the whole image can be mathematically seen as a **convolution**.

So, while we slide the filter, at each location of the filter, the output is composed of the dot product between the values of the filter and the corresponding location in the image (both vectorised), where the values of the filters are the weigths that we want to learn, and they remain constant across the sliding. If our filter is a $10 \times 10$ matrix, we have only 100 weights to learn by layer (plus one for the offset). Exactly as for feedforward neural networks, then the dot product is passed through an activation function, here typically the `ReLU` function ($max(0,x)$) rather than `tanh`:

![Convolutional filter](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/convolutionalFilter.png)

### Example

For example, given an image $x =
\begin{bmatrix}
1 & 1 & 2 & 1 & 1 \\
3 & 1 & 4 & 1 & 1 \\
1 & 3 & 1 & 2 & 2 \\
1 & 2 & 1 & 1 & 1 \\
1 & 1 & 2 & 1 & 1 \\
\end{bmatrix}$ and filter weights $w =
\begin{bmatrix}
 1 & -2 & 0 \\
 1 &  0 & 1 \\
-1 &  1 & 0 \\
\end{bmatrix}$, then the output of the filter $z$ would be $\begin{bmatrix}
 8 & -3 & 6 \\
 4 &  -3 & 5 \\
-3 &  5 & -2 \\
\end{bmatrix}$. For example, the element of this matrix $z_{2,3} = 5$ is the result of the sum of the scalar multiplication between $x^\prime = \begin{bmatrix}
 4 &  1 & 1 \\
 1 &  2 & 2 \\
 1 &  1 & 1 \\
\end{bmatrix}$ and $w$.

Finally, the output of the layer would be (using ReLU) $\begin{bmatrix}
 8 & 0 & 6 \\
 4 &  0 & 5 \\
 0 &  5 & 0 \\
\end{bmatrix}$.

We can run the following snippet to make the above computations:

```@repl 0401-nn.jl
ReLU(x) = max(0,x)
x = [1 1 2 1 1;
        3 1 4 1 1;
        1 3 1 2 2;
        1 2 1 1 1;
        1 1 2 1 1]
w = [ 1 -2  0;
        1  0  1;
        -1  1  0]
(xr,xc) = size(x)
(wr,wc) = size(w)
z = [sum(x[r:r+wr-1,c:c+wc-1] .* w) for c in 1:xc-wc+1 for r in 1:xr-wr+1] # Julia is column major
u = ReLU.(z)
final = reshape(u, xr-wr+1, xc-wc+1)
```


You can notice that, applying the filter, we obtain a dimensionality reduction. This reduction depends on both the dimension of the filter and the stride (sliding step). In order to avoid this, a padding of one or more row/column can be applied to the image in order to preserve in the output the same dimension of the input (in the above example a padding of one row and one column on both sides would suffice). Typically the padded cells are given a value of zero so not to contribute anything when they are included in the dot product computed by the  filter.

To determine the spatial size in output of a filter ($O$), given the input size ($I$), the filter size ($F$), the stride ($S$) and the eventual padding ($P$) we can use the following simple formula:

$O_d = 1 + (I_d+2*P_d-F_d)/S$ 

From it, we can also find the padding needed to obtain a certain output size as:
$P_d = ((O_d-1)S_d-I_d+F_d)/2$

Where the $d$ index accounts for the (extremelly unusual) case where one of the parameter is not a square matrix, so that for example an image has different vertical and horizzontal resolutions.


Because the weights of the filters are the same, it doesn't really matter where the object is learned, in which part of the image. With convolutional layers, we have _translational invariance_ as the same filter is passed over the entire image. Therefore, it will detect the patterns regardless of their location.

Still, it is often convenient to operate some **data augmentation** to the training set, that is to add slightly modified images (rotated, mirrored..) in order to improve this translational invariance.






### Considering multiple filters per layer

Typically, one single layer is formed by applying multiple filters, not just one. This is because we want to learn different kind of features. For example in an image one filter will specialize to catch vertical lines, an other obliques ones, and maybe an other filter different colours.

![Set of different convolutional filters outputs](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/convolutionalLayerOutputs.png)

Convolutional filters outputs on the first layer (filters are of size 11x11x3 and are applied across input images of size 224x224x3). Source: [Krizhevsky et Oth. (2012), "ImageNet Classification with Deep Convolutional Neural Networks"](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

So in each layer we map the output of the previous layer (or the original image in case of the first layer) into multiple feature maps where each feature map is generated by a little weight matrix, the filter, that defines the little classifier that's run through the original image to get the associated feature map. Each of these feature maps defines a channel for information and we can represent as a third dimension to form a "volume", where the depth is given by the different filters:

![Convolutional layer](https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsSources/04_-_NN_-_Neural_Networks/imgs/convolutionalLayer.png)

In the image above the input layer has size (4,4,4) and the output layers has size (3,3,2), i.e. 2 "independent" filters of size (3,3).

For square layers, each filter has $F^2 \times D_{l-1} + 1$ parameters where $D_{l-1}$ is the dimensions ("depth") of the previous layer, so the total number of parameters per layer is $(F^2 \times D_{l-1} + 1) * D_l$.

For computational reasons the number of filters per layer $D_l$ is normally a power of 2. 

This representation allows to remain consistent with the input that can as well be represented as a volume. For images the depth is usualy given by 3 layers representing the values in terms of RGB colours.

### Pool layers

A further way  to improve translational invariance, but also have some dimensionality reduction, it called **pooling** and is adding a layer with a filter whose output is the `max` of the corresponding area in the input (or, more rarelly, the average). Note that this layer would have no weights to learn!
With pooling we contribute to start separating what is in the image from where it is in the image, that is pooling does a fine-scale, local translational invariance, while convolution does more a large-scale one.

Keeping the output of the above example as input, a pooling layer with a $2 \times 2$ filter and a stride of 1 would result in $\begin{bmatrix}
 8 & 6 \\
 5 & 5 \\
\end{bmatrix}$.


### Convolutional networks conclusions

We can then combine these convolution, looking for features, and pooling,
compressing the image a little bit, forgetting the information of where things are, but maintaining what is there.

In a typical CNN, these convolutional and pooling layers are repeated several times, where the initial few layers typically would capture the simpler and smaller features, whereas the later layers would use information from these low-level features to identify more complex and sophisticated features, like characterisations of a scene. The learned weights would hence specialise across the layers in a sequence like  edges -> simple parts-> parts -> objects -> scenes.

These layers are finally followed by some "normal", "fully connected" layers (like in "normal" feed-forward neural networks) and a final `softmax` layer indicating the probability that each image represents one of the possible categories (there could be thousand of them).

The best network implementation are tested in so called "competitions", like the yearly ImageNet context.

Note that we can train this networks exactly like for feedformard NN, defining a loss function and finding the weights that minimise the loss function. In particular we can apply the stochastic gradient descendent algorithm (with a few tricks based on getting pairs of image and the corresponding label), where the gradient with respect to the various parameters (weights) is obtained by backpropagation.