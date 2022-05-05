# [Course Presentation](@id course_presentation)


## What will you get from this course

This course aims to provide a gentle introduction to Julia, a modern, expressive and efficient programming language, and to introduce the main concepts around machine learning, together with some algorithms and practical implementations of machine learning workflows.

In particular, by attending this course you will receive : 
1. An operational knowledge of a modern and efficient general-purpose language, with a focus on numerical and scientific computation;
2. An exposure to several specific but commonly used tools in multiple scientific areas (data wrangling and visualisation, constrained optimisation, ...)
3. An overview of modern tools for scientific collaboration and software quality, such as version control systems and best practices to obtain replicable results.
4. An introduction to machine learning approaches: scopes, terminology, typologies, workflow organisation
5. An in-deep introduction to some specific machine learning algorithms for classification and regression (perceptron, neural networks, random forests..)
6. Experience in employing Machine Learning workflows to specific cases on multiple domains

## How to attend the course

The course is multi-channel, with these pages complemented with youtube videos (see [Program](@ref course_program)), quizzes and exercises. Thanks to projects such as [Literate.jl](https://github.com/fredrikekre/Literate.jl) and [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) the source of most of these pages are runnable valid Julia files.

**To fully master the subject, nothing is better than cloning [the repository on GitHub](/lessonsSources/00_-_INTRO_-_Introduction_julia_ml/assets/imgs/zenodo.6509909.svg) and running these pages by yourself!**
To execute yourself the code discussed in the videos, run the code in the `lessonsSource` folder.
Other resources used in the course (in particular the examples of the introduction and the exercises) are located under the `lessonsMaterial` folder.

While the content of this course may also be used in specific academic courses with a fixed calendar, you are free to complete the course at your pace. You don't even need to log in. This however means that there is no way to track your progress or the outcomes of the quizzes.

Note that there will be a bit of incongruence in the terminology used for "units". Sometimes I call them "lessons", especially at the beginning of the course, then it ended up that the actual content was much bigger than a standard "lesson", so I started to use the word "unit".
Anyhow, the course is organised in units/lessons (e.g. `JULIA1`). Each unit/lesson has its own Julia environment and it is organised in several _segments_. These are the individual pages on this site and correspond to a single file in the GitHub repository. Then each segment is divided into multiple videos, which I call _parts_.


## [Course authors](@id course_authors)

-----

```@raw html
 <p style ="font-weigth: bold;"><img src="assets/imgs/photo_antonello_lobianco.png" alt="Antonello Lobianco" width="100" style="float: left; margin-right: 15px;"> 
 Antonello Lobianco</p>
 <p> <a href="https://orcid.org/0000-0002-1534-8697">OrcID</a> - <a href="https://scholar.google.com/citations?user=8DSfpVUAAAAJ">Google Schoolar</a> - <a href="https://github.com/sylvaticus/">GitHub</a> - <a href="https://lobianco.org/antonello">Personal site</a>
 </p>
 <p style="clear:both;">
```


I am a Forest and Natural Resources Economist @ AgroParisTech, a French "Grand Ecole" (sort of polytechnic university) in the life science domain and affiliated to BETA, the _Bureau d'Économie Théorique et Appliquée_, which brings together most of the economists located in the _Grand-Est_ region of France.
My main interest is in exploring the interplay between the biophysical layers in the forest sector (climate change, forest dynamics) and the socio-economic ones (individual forest owners' behaviours, timber markets "behaviours") in order to better understand the space of actions that society has to maximise social benefits in a sustainable way. For that, I design, develop and use bio-economic simulation models of the forest sector.
I started moving my research models to Julia in 2017 and in 2019 I wrote the book "[Julia Quick Syntax Reference: A Pocket Guide for Data Science Programming](https://doi.org/10.1007/978-1-4842-5190-4)" (Apress, 2019)

While this course is in English, English is not my native language... please, be understanding !

**Acknowledgements**: The development of this course at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](https://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” program (ANR 11 – LABX-0002-01).

-----

## How to cite the course

Please cite as :

Antonello Lobianco (2022), _Introduction to Scientific Programming and Machine
Learning with Julia (SPMLJ)_, doi: [10.5281/zenodo.6425030](https://doi.org/10.5281/zenodo.6425030)

[![DOI](assets/imgs/zenodo.6509909.png)](https://zenodo.org/badge/latestdoi/429458515)




## [How to contribute to the course](@id contribution_guidelines)

Contributions are really welcome! From simple, minor editing and bug fixes (by making new comments or, better, pull requests) to introducing new arguments along with the course domain.

The only 3 conditions are :
1. prerequisites for the students are kept at the lowest possible level. If one concept can be fully expressed using a simple, trivial "2+2" method, this should be always preferred to using more advanced mathematical concepts;
2. a practical implementation of the concepts in a working Julia file is proposed to the students and the documentation builds on top of that file (this site is built using Literate.jl that transforms the `.jl` file to `.md`, and then Documenter.jl that transforms the `.md` file to html);
3. some sort of interactive tools are proposed to the students, like the quizzes or practical exercises.

While the production of videos for your content is welcome, this is not strictly necessary.
