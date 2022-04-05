
# Course Presentation


## Objectives 

This course aims to provide a gentle introduction to Julia, a modern, expressive and efficient programming language, and introduce the main concepts around machine learning, together with some algorithms and practical implementations of machine learning workflows.

In particular the objectives are : 
1. Supply students with an operational knowledge of a modern and efficient general-purpose language that can be employed for the implementation of their research daily activities;
2. Present several specific but commonly used tools in multiple scientific areas (data transformation, optimisation,...)
3. Introducing students to modern tools for scientific collaboration and software quality, such as version control systems and best practices to obtain replicable results.
4. Introduce machine learning approaches: scopes, terminology, typologies, workflow organisation
5. Introduce some specific machine learning algorithms for classification and regression

## How to attend the course

The course is multi-channel, with these pages complemented with youtube videos (see [`Program`](@ref) ), quizzes and exercises. Thanks to projects as [Literate.jl](https://github.com/fredrikekre/Literate.jl) and [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) the source of most of these pages are runnable valid Julia files.

**To fully master the subject, nothing is better than cloning [the repository on GitHub](https://github.com/sylvaticus/SPMLJ) and run these pages by yourself!**
To execute yourself the code discussed in the videos run the code in the `lessonsSource` folder.
Other resources used in the course (in particular the examples of the introduciton and the exercises) are located under the `lessonsMaterial` folder.

While the content of this course may also be used in specific academic courses with a fixed calendar, you are free to complete the course at your pace. You don't even need to log-in. This however means that there is no way to track your progress or the outcomes of the quizes.

Note that there will be a bit of incongruence in the terminology used for "units". Sometimes I call them "lessons", especially at the beginning of the course, then it ended up that the actual content was much bigger than a standard "lesson", so I started to use the word "unit".
Any-how, the course is organised in units/lessons (e.g. `JULIA1`). Each unit/lesson has its own Julia environment and it is organised in several _segments_. These are the individual pages on this site and correspond to a single file in the github repository. Then each segment is divised in multiple videos, that I call _parts_.


## Course authors


```@raw html
 <img src="assets/imgs/photo_antonello_lobianco.png" alt="Antonello Lobianco" width="150"> 
```


Antonello Lobianco

[OrcID](https://orcid.org/0000-0002-1534-8697) - [Google Schoolar](https://scholar.google.com/citations?user=8DSfpVUAAAAJ) - [GitHub]( https://github.com/sylvaticus/) - [Personal site](https://lobianco.org/antonello)

I am a Forest and Natural Resources Economist @ AgroParisTech, a French "Grand ecole" (sort of polytecnic university) in the life science domain and affiliated to BETA, the _Bureau d'Économie Théorique et Appliquée_, which brings together most of the economists located in the _Grand-Est_ region of France.
My main interest is in exploring the interplay between the biophysical layers in the forest sector (climate change, forest dynamics) and the socio-economic ones (individual forest owners behaviours, timber markets "behaviours") in order to better understand the space of actions that society has to maximise social benefits in a sustenible way. For that I design, develop and use bio-economic simulation models of the forest sector.

While this course is in English, English is not my native language... please, be understanding !


## How to contribute to the course

Contributions are really welcome! From simple, minor editing and bug-fixes (by making new comments or, better, pull requests) to introducing new arguments along the course domain.

The only 3 conditions are that:
1. the prerequisites for the students are kept at the lowest possible level. If one concept can be fully expressed using a simple, trivial "2+2" method, this should be always preferred to using more advanced mathematical concepts;
2. a practical implementation of the concepts in a working Julia file is proposed to the students and the documentation builds on top of that file (this site is built using Literate.jl that transforms the `.jl` file to `.md`, and then Documenter.jl that tranforms the `.md` file to html);
3. some sort of interactive tools are proposed to the students, like the quizzes or  practical exercises.

While the production of videos for your content is welcome, this is not strictly necessary.



```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```