# IntroSPMLJuliaCourse
Introduction to Scientific Programming and Machine Learning with Julia

https://github.com/sylvaticus/IntroSPMLJuliaCourse

STATUS: Up to 1.3F

Objectives: 
1. Supply students with an operational knowledge of a modern and efficient general-purpose language that can be employed for the implementation of their research daily activities;
2. Present several specific but commonly used tools in multiple scientific areas (data transformation, optimisation, symbolic calculation,...)
3. Introducing students to modern tools for scientific collaboration and software quality, such as version control systems and best practices to obtain replicable results.
4. Introduce machine learning approaches: scopes, terminology, typologies, workflow organisation
5. Introduce some specific machine learning algorithms for classification and regression

Resources provided:
- julia scripts
- the same script with the output
- videos divided in "small" segments and parts
- interactive quizzes between each segment or part (at time of writing quizzes are available only for students with access to the specific elearning platform used for the course, I am looking for a way to provide them to the general public)

To execute yourself the code in the videos in the lessons run the code in the `lessonsSource` folder.
Other resources used in the course (in particualr the examples of the Keek-off meeting) are located under the `lessonsMaterial` folder.

Videos (hosted on YouTube):

### 00 KOM: Keek-off meeting (2h:44:54)
- [Course introduction](https://www.youtube.com/watch?v=2tM5oIvOQnU&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=1) (16:11)
- [Julia overview](https://www.youtube.com/watch?v=uW8iyTjSaJk&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=2&t=1149s) (36:25)
- [ML Terminology](https://www.youtube.com/watch?v=l9ls2yssKiE&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=3) (21:19)
- [A first ML Example](https://www.youtube.com/watch?v=SclPPNvYEAI&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=4&t=18s) (7:00)
- [ML application areas](https://www.youtube.com/watch?v=bIApDXIhm1k&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=5&t=32s) (14:24)
- Hands on (42:09)
  - [Part A](https://www.youtube.com/watch?v=kT9Vm8Ov6qo&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=6&t=15s) (20:15)
  - [Part B](https://www.youtube.com/watch?v=51AwIJzxtgw&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=7&t=1135s) (21:54)
- [Pkg and the environments](https://www.youtube.com/watch?v=_qTLSrk1ICA&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=8&t=517s) (20:56)
- [Further ML Examples](https://www.youtube.com/watch?v=9Kni5XkQV5M&list=PLDIpPSqVuMmLGUNGMXL2eO2pqKlzdPfxa&index=9) (6:34)

### 01 JULIA1: Basic Julia programming (2h:53:15)
- Basic syntax elements (46:45)
  - [Part A - Introduction and setp of the environment](https://www.youtube.com/watch?v=fRv3vAmzHS8&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=1) (8:37)
  - [Part B - Comments, code organsation, unicode support, broadcasting](https://www.youtube.com/watch?v=1SVA6woAq18&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=2&pp=sAQB) (12:04)
  - [Part C - Math operators, quotation marks](https://www.youtube.com/watch?v=1AVb-92QmPg&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=3&pp=sAQB) (6:49)
  - [Part D - Missing values](https://www.youtube.com/watch?v=UGlJlH1BbjM&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=4&pp=sAQB) (10:04)
  - [Part E - Stochasticity in programming](https://www.youtube.com/watch?v=Hi_a7YWA_j8&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=5&pp=sAQB) (9:14)
- Types and objects (26:38)
  - [Part A - Types, objects, variables and operators](https://www.youtube.com/watch?v=tOqojCvpAuE&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=6&pp=sAQB) (13:05)
  - [Part B - Object mutability and effects on copying objects](https://www.youtube.com/watch?v=yeaTNPKqRTo&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=7&pp=sAQB) (13:34)      
- Predefined types (1h:39:50)
  - [Part A - Primitive types, char and strings](https://www.youtube.com/watch?v=kEb50RK1sK0&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=8&pp=sAQB) (9:37)
  - [Part B - One dimensional arrays](https://www.youtube.com/watch?v=4nR1rI8_hug&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=9&pp=sAQB) (30:42)
  - [Part C - Multidimensional arrays](https://www.youtube.com/watch?v=WJcikzhIr7Y&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=10&pp=sAQB) (23:37)
  - [Part D - Tuples and named tuples](https://www.youtube.com/watch?v=79kRlC5dbAo&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=11&pp=sAQB) (7:50)
  - [Part E - Dictionaries and sets](https://www.youtube.com/watch?v=KwY_dfvzByk&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=12&pp=sAQB) (8:57)
  - [Part F - Date and times](https://www.youtube.com/watch?v=y4Ty2Wx_lC4&list=PLDIpPSqVuMmK1poGUS7nuAXXILxHxmV2O&index=13&pp=sAQB) (19:11)
