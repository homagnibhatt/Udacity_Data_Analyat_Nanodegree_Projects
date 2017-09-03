Titanic Data Visualization Project

On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. 
Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, upper-class etc. 
The purpose of this project is to identify the various groups who were likely to survive. 
This dataset was obtained from the Kaggle website. The training dataset was used.

Description of the data

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

Data Cleaning

Although the dataset is fairly cleaned, some preliminary cleaning was performed using R. The following steps were taken:

•	Removing rows with NA and blank values so that a complete dataset was used.

•	Since we are interested in looking into the survivors and the factors which influence it, we will only use data where Survived = 1 i.e. only those data points where the passenger survived.

•	Now we will make a preliminary observation in the dataset. Hence, we will look into only a limited number of variable and its interplay with survival.

•	For this reason we only use the following variables: Survived, PClass, Sex, Age, Fare, Embarked

Now we will dive into the visualization.

