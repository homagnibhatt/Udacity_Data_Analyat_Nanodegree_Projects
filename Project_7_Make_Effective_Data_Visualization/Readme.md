#Titanic Data Visualization Project

On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. 
Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, upper-class etc. 
This dataset was obtained from the Kaggle website. The training dataset was used.

#Description of the data

Below is a brief information about each columns of the dataset:
1.	PassengerId: An unique index for passenger rows. It starts from 1 for first row and increments by 1 for every new rows.
2.	Survived: Shows if the passenger survived or not. 1 stands for survived and 0 stands for not survived.
3.	Pclass: Ticket class. 1 stands for First class ticket. 2 stands for Second class ticket. 3 stands for Third class ticket.
4.	Name: Passenger's name. Name also contain title. "Mr" for man. "Mrs" for woman. "Miss" for girl. "Master" for boy.
5.	Sex: Passenger's sex. It's either Male or Female.
6.	Age: Passenger's age. "NaN" values in this column indicates that the age of that particular passenger has not been recorded.
7.	SibSp: Number of siblings or spouses travelling with each passenger.
8.	Parch: Number of parents of children travelling with each passenger.
9.	Ticket: Ticket number.
10.	Fare: How much money the passenger has paid for the travel journey.
11.	Cabin: Cabin number of the passenger. "NaN" values in this column indicates that the cabin number of that particular passenger has not been recorded.
12.	Embarked: Port from where the particular passenger was embarked/boarded.

#Data Cleaning

Although the dataset is fairly cleaned, some preliminary cleaning was performed using R. The following steps were taken:
1.	Removing rows with NA and blank values so that a complete dataset was used.
2.	Since we are interested in looking into the survivors and the factors which influence it, we will only use data where Survived = 1 i.e. only those data points where the passenger survived.
3.	Now we will make a preliminary observation in the dataset. Hence, we will look into only a limited number of variable and its interplay with survival.
4.	For this reason we only use the following variables: Survived, PClass, Sex, Age, Fare, Embarked

A stacked percent barchart was used to encode the data.

The following design was used:

1.	The survived was plotted as a percent y axis.
2.	The x axis represented the Passenger class broken by the sex of the survivors.
3.	The bars were stacked with the Ports that the passengers embarked from.

The following points can bee seen:
1. Within the survivors, First class passengers have the highest number of survivors.	
2. In all classes, the women have better suvivability than men accross all ages. 
3. Interestingly, highest number of survivors are from Southampton followed by Cherbourg and only marginal survivors from Queenstown accross all classes. 
4.Hence, we can infer that there is a greater chance for female passengers from Southampton belonging to high socio-economic status to have survived the Titanic disaster.

Some more changes were done to improve the cart which was incorporated in index_final.html

The following changes were done:

1. The chart type was now a storyboard type with the animating feature being the Age_Bucket (creating by grouping the age into defined groups in the R script).
2. The Age_Bucket was introduce another crucial element Age, with respect to Titanic survivors and animating it helped to gain a very nice overview of the impact of Class, Sex, Count and Age factors in the survival of the Titanic passengers.

Refences:
dimplejs.org
