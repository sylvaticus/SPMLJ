# 0501 - Decision trees based algorithms
**[Draft]**



![GuessWho](imgs/guessWho.jpg)

Have you ever played with the board game _Guess Who?_ when you were a child ? The game objective is to find a particular character by asking questions related to its characteristics, like the sex, the colour of the hairs or the eyes, etc...

Decisions trees are the collections of "questions" that you pose, relative to some data characteristics (the _features_, in ML jargon), to arrive determining some other unknown characteristics (the _labels_).
In the board game the unknown characteristics is the name of the character, and it is unique across the board.
But more in general it can be a characteristic shared by many individuals. For example we could play _Guess Who?_ with the objective to determine the sex of the character (without of course the possibility to ask about it).

It then become obvious that the "best" question is those that can split our records the most evenly possible in groups based on the desired feature. In the example above asking if the character has barb, or earrings,  would likelly be good questions. 

Decision trees algorithms are algorithms that indeed learn from the data to ask the "best" questions to arrive to features in the shortest possible term, i.e. by posing the less amount of questions.

Decision trees are very quick and efficient methods to predict a label, either for classification or for regression, but are highly sensitive to the specific data on which they are trained.
Random forests are _ensemble_ of decision trees, each one trained on slighly different data, as the _N_ records avaialble for training are sampled with replacement _N_ times, so some records may appear multiple times, some others are not used for a particular tree (this tecnique is called _boostrapping_).
Also, each decision trees consider (again, randomly) only a subset of features (dimensions) from the given data during training.

In a Random Forest, the individual decision trees outputs are then aggregated and weighted to provide an output with much less variance than individual trees (the _boostrap_ plus _aggregation_ takes the name of _bagging_).

The trees that are not been used (sampled) by each decision tree can be used to provide a so called _out-of-bag_ estimate of the validation error without the need to use a separate validation set. Cool, isn't it ?!

`BetaML` supports Decision Trees and Random forest respectivelly with the [`buildTree(x,y;...)`](https://sylvaticus.github.io/BetaML.jl/stable/Trees.html#BetaML.Trees.buildTree-Union{Tuple{Ty},%20Tuple{Any,%20AbstractVector{Ty}}}%20where%20Ty) and [`buildForest(x,y,nTrees;...)`](https://sylvaticus.github.io/BetaML.jl/stable/Trees.html#BetaML.Trees.buildForest-Union{Tuple{Ty},%20Tuple{Any,%20AbstractVector{Ty}},%20Tuple{Any,%20AbstractVector{Ty},%20Any}}%20where%20Ty) functions that perform the training.
Their outputs can then be used to make predictions with `predict(treeOrForest,x)`.

Note that while faster implementations of DT/RF exists (for ex. the [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) package), BetaML RF/DT are quite unique in the way they do not assume any order in the feature dimensions and work straight away, without the need of any preprocessing, with almost any kind of data. Continuous, integer, categorical, custom types... even missing data is supported.
Similarly, the kind of job (regression/classification) is automatically determined by the type of the labels, with the otion to force a classification job even for numerical labels.

While RF generalise well, still a few hyperparameters are provided in order to improve regularisation:
- `nTrees`: Number of trees in the forest [def: `30`]
- `maxDepth`: The maximum depth the tree is allowed to reach. When this is reached the node is forced to become a leaf [def: `N`, i.e. no limits]
- `minGain`: The minimum information gain to allow for a node's partition [def: `0`]
- `minRecords`:  The minimum number of records a node must holds to consider for a partition of it [def: `2`]
- `maxFeatures`: The maximum number of (random) features to consider at each partitioning [def: `√D`,]
- `splittingCriterion`: Either `gini`, `entropy` or `variance`