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


#** Data Visualization: Titanic Data Visualization ** #

by Homagni Bhattacharjee, as Project 7 of Udacity's Data Analyst Nanodegree.

##** Summary ** ##

This project charts a storyboard stacked bar chart. It shows the following:
1.The animating feature is the Age_Bucket (creating by grouping the age into defined groups in the R script).
2. The chart gives a nice overview of the impact of Class, Sex, Count and Age factors in the survival of the Titanic passengers.
3. Also it provides the port embarked from as a fill factor of the chart, enabling us to identify the number of survivors from eact city in each category.

##** Design ** ##

###** Exploratory Data Analysis and Cleaning (R) **##

I downloaded the data from Kaggle - Machine Learning from Disaster, selecting a train data set which already had basic finding & doesn't need extensive data wrangling or transformation.
Although the dataset is fairly cleaned, some preliminary cleaning was performed using R. The following steps were taken:
1.	Removing rows with NA and blank values so that a complete dataset was used.
2.	Since we are interested in looking into the survivors and the factors which influence it, we will only use data where Survived = 1 i.e. only those data points where the passenger survived.
3.	Here, we make a preliminary observation in the dataset. Hence, we will look into only a limited number of variable and its interplay with survival.
4.	For this reason we only use the following variables: Survived, PClass, Sex, Age, Fare, Embarked

###** Data Visualization (dimple.js) ** ##

I decided to use solely and dimple.js as it would be sufficient for this task:

I considered using multiple chart types (scatter, line chart, bubble chart, bar chart, etc.), color each line separately to test if this is a good way to visualize & stress on important point. I re-evaluated different chart type by tweaking few line of code and confirm my initial assumption, a stacked bar chart is already sufficient to dislay data characteristic. The first version is drawn from index-initial.html. This initial iteration can be viewed at index-initial.html, or below:

![First Chart](https://raw.githubusercontent.com/tommyly2010/Udacity-Data-Analyst-Nanodegree/master/p6 - Data Visualization/img/image-initial.png)

###** Feedback **##

I gathered feedback from 3 different people people and tried to follow Udacity questions guideline and here is the abridged responses.

**Interview #1**

> Your chart was a bit messy & no clear headlines & legend. If there is some explanation like a bold headline then these charts would make sense. The insights is not so much of a surprise and if you just create a small tweak in the legend, x-axis & y-axis then this would looks good. The data clearly favors your initial hypothesis, women children & elders are prioritized to board the baot first.

**Interview #2**

The chart is interactive, that's nice. But why the second chart is revert and not in-line with the other chart. Switch x-axis & y-axis with each other and I think the chart will looks much better. By the way, the first chart to split between classes is cool but I think you can make it even better by combining gender & classes to see if there's different behavior in different classes. Broadly speaking, the chart looks intuitive & only needs a few small tweak.

**Interview #3**

The second chart looks a bit weird and too much junk information, there's no need to include different age in different age bracket like that. There's not much information to show. And for the first chart, split the column into two, a stacked-bar wouldn't be necessary. Also, you also needs to clean up the headline & make clear of the axis, what is PClass? Can you makes it a bit clearer. Overall, this chart is straightforward.

###** Post-feedback Design ** ###

Following the feedback from the 3 interviews, I implemented the following changes:

I separate man & women from the first chart.
I added careful chart title & clearly labeled axis title.
I flipped the chart from horizontal bar chart to vertical bart chart.
I remove the individual age & only shows aggergrate age group.
I intended to add few special effect (highlight a chart when mouseover) but this would not be necessary.
I switched from Number of Survival to Survival Rates since the amount of passengers in each class/ages group is not similar.
Final rendition of the data visualization is shown below:

![Final Chart](https://raw.githubusercontent.com/tommyly2010/Udacity-Data-Analyst-Nanodegree/master/p6 - Data Visualization/img/image-final.png)

##** Resources **##

dimple.js Documentation
Data Visualization and D3.js (Udacity)
mbostock's blocks
Dimple homepage

##** Data **##

train.csv: original downloaded dataset with minor cleaning for dimple.js implementation.
