# IntroSPMLJuliaCourse
Introduction to Scientific Programming and Machine Learning with Julia

https://github.com/sylvaticus/IntroSPMLJuliaCourse

STATUS: Work in progress. The course is still being worked on (2021.03.04: the part on Julia is completed ;-) )

Objectives: 
1. Supply students with an operational knowledge of a modern and efficient general-purpose language that can be employed for the implementation of their research daily activities;
2. Present several specific but commonly used tools in multiple scientific areas (data transformation, optimisation, symbolic calculation,...)
3. Introducing students to modern tools for scientific collaboration and software quality, such as version control systems and best practices to obtain replicable results.
4. Introduce machine learning approaches: scopes, terminology, typologies, workflow organisation
5. Introduce some specific machine learning algorithms for classification and regression

Resources provided:
- julia scripts
- the same script with the output
  - [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sylvaticus.github.io/IntroSPMLJuliaCourse/dev)
- videos divided in "small" segments and parts
- interactive quizzes between the different segments or parts (at time of writing quizzes are available only for students with access to the specific elearning platform used for the course, the plan it to integrate them in the compiled documentation for the general public, e.g. using [QuizQuestions.jl](https://github.com/jverzani/QuizQuestions.jl))

To execute yourself the code in the videos in the lessons run the code in the `lessonsSource` folder.
Other resources used in the course (in particualr the examples of the Keek-off meeting) are located under the `lessonsMaterial` folder.

Videos (hosted on YouTube):

### 00 KOM: Keek-off meeting (2h:44:54)
- [Course introduction](https://www.youtube.com/watch?v=2tM5oIvOQnU&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=1) (16:11)
- [Julia overview](https://www.youtube.com/watch?v=uW8iyTjSaJk&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=2) (36:25)
- [ML Terminology](https://www.youtube.com/watch?v=l9ls2yssKiE&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=3) (21:19)
- [A first ML Example](https://www.youtube.com/watch?v=SclPPNvYEAI&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=4) (7:00)
- [ML application areas](https://www.youtube.com/watch?v=bIApDXIhm1k&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=5) (14:24)
- Hands on (42:09)
  - [Part A](https://www.youtube.com/watch?v=kT9Vm8Ov6qo&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=6) (20:15)
  - [Part B](https://www.youtube.com/watch?v=51AwIJzxtgw&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=7) (21:54)
- [Pkgs, modules and environments](https://www.youtube.com/watch?v=_qTLSrk1ICA&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=8) (20:56)
- [Further ML Examples](https://www.youtube.com/watch?v=9Kni5XkQV5M&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=9) (6:34)

### 01 JULIA1: Basic Julia programming (5h:52:33)
- Basic syntax elements (46:45)
  - [Part A - Introduction and setup of the environment](https://www.youtube.com/watch?v=fRv3vAmzHS8&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=1) (8:37)
  - [Part B - Comments, code organsation, Unicode support, broadcasting](https://www.youtube.com/watch?v=1SVA6woAq18&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=2) (12:04)
  - [Part C - Math operators, quotation marks](https://www.youtube.com/watch?v=1AVb-92QmPg&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=3) (6:49)
  - [Part D - Missing values](https://www.youtube.com/watch?v=UGlJlH1BbjM&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=4) (10:04)
  - [Part E - Stochasticity in programming](https://www.youtube.com/watch?v=Hi_a7YWA_j8&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=5) (9:14)
- Types and objects (26:38)
  - [Part A - Types, objects, variables and operators](https://www.youtube.com/watch?v=tOqojCvpAuE&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=6) (13:05)
  - [Part B - Object mutability and effects on copying objects](https://www.youtube.com/watch?v=yeaTNPKqRTo&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=7) (13:34)      
- Predefined types (1h:39:50)
  - [Part A - Primitive types, char and strings](https://www.youtube.com/watch?v=kEb50RK1sK0&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=8) (9:37)
  - [Part B - One dimensional arrays](https://www.youtube.com/watch?v=4nR1rI8_hug&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=9) (30:42)
  - [Part C - Multidimensional arrays](https://www.youtube.com/watch?v=WJcikzhIr7Y&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=10) (23:37)
  - [Part D - Tuples and named tuples](https://www.youtube.com/watch?v=79kRlC5dbAo&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=11) (7:50)
  - [Part E - Dictionaries and sets](https://www.youtube.com/watch?v=KwY_dfvzByk&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=12) (8:57)
  - [Part F - Date and times](https://www.youtube.com/watch?v=y4Ty2Wx_lC4&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=13) (19:11)
- Control flow and functions (44:47)
  - [Part A - Variables scope](https://www.youtube.com/watch?v=S8HcWitIRZg&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=14) (9:47)
  - [Part B - Loops and conditional statements](https://www.youtube.com/watch?v=DGh_5aNKghI&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=15) (9:2)   
  - [Part C - Functions](https://www.youtube.com/watch?v=oejR6LKvFXY&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=16) (25:57)
- Custom Types (40:02)
  - [Part A - Types of types, composite types](https://www.youtube.com/watch?v=-tuVyAixoXE&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=17) (17:56)
  - [Part B - Parametric types](https://www.youtube.com/watch?v=UCybSmhURIE&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=18) (7:37)
  - [Part C - Inheritance and composition OO paradigms](https://www.youtube.com/watch?v=gv8ZIThsHTo&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=19) (14:28)  
- Further Topics (1:34:30)
  - [Part A - Metaprogramming and macros](https://www.youtube.com/watch?v=Q3Fx6pFLCFk&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=20) (23:46)
  - [Part B - Interoperability with other languages](https://www.youtube.com/watch?v=xK_Ug2gtQvU&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=21) (23:6)
  - [Part C - Performances and errors: profiling, debugging, introspection and exceptions](https://www.youtube.com/watch?v=vg8v_6oX2DM&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=22)(27:33)
  - [Part D - Parallel computation: multithreading, multiprocessing](https://www.youtube.com/watch?v=L849oXXCXFM&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=23) (20:3)


### 02 JULIA2: Scientific Julia programming (2h:39:34)
- Data Wrangling (1h:31:15)
  - [Part A - Introduction and data import](https://www.youtube.com/watch?v=o3TiuZlxMKI&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=1) (18:28)
  - [Part B - Getting insights of the data](https://www.youtube.com/watch?v=_iQh9dddIUY&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=2) (25:26)
  - [Part C - Edit data and dataframe structure](https://www.youtube.com/watch?v=qhhT-Ckyvag&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=3) (23:40)
  - [Part D - Pivot, Split-Apply-Combine and data export](https://www.youtube.com/watch?v=w1KcD8sJLok&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=4) (23:39)
- Further topics (1h:08:19)
  - [Part A - Plotting](https://www.youtube.com/watch?v=b-WkKMLd-Ws&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=5) (15:18)
  - [Part B - Probability distributions and data fitting](https://www.youtube.com/watch?v=cfNXk2eiIFE&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=6) (11:55)
  - [Part C - Constrained optimisation, the transport problem](https://www.youtube.com/watch?v=vxo7nOpnFSs&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=7) (24:59)
  - [Part D - Nonlinear constrained optimisation, the optimal portfolio allocation](https://www.youtube.com/watch?v=_ypOlSwCC7U&list=PLDIpPSqVuMmI4Dhiekw2y1wakzsaMSJVG&index=8) (16:04)
 