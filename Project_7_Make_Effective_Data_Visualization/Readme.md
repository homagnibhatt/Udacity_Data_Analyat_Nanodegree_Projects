**Data Visualization: Titanic Data Visualization** 

by Homagni Bhattacharjee, as Project 7 of Udacity's Data Analyst Nanodegree.

**Summary**

This project charts a storyboard stacked bar chart. It shows the following:
1.The animating feature is the Age_Bucket (creating by grouping the age into defined groups in the R script).
2. The chart gives a nice overview of the impact of Class, Sex, Count and Age factors in the survival of the Titanic passengers.
3. Also it provides the port embarked from as a fill factor of the chart, enabling us to identify the number of survivors from eact city in each category.

**Design**

**Exploratory Data Analysis and Cleaning (R)**

I downloaded the data from Kaggle - Machine Learning from Disaster, selecting a train data set which already had basic finding & doesn't need extensive data wrangling or transformation.
Although the dataset is fairly cleaned, some preliminary cleaning was performed using R. The following steps were taken:
1.	Removing rows with NA and blank values so that a complete dataset was used.
2.	Since we are interested in looking into the survivors and the factors which influence it, we will only use data where Survived = 1 i.e. only those data points where the passenger survived.
3.	Here, we make a preliminary observation in the dataset. Hence, we will look into only a limited number of variable and its interplay with survival.
4.	For this reason we only use the following variables: Survived, PClass, Sex, Age, Fare, Embarked

**Data Visualization (dimple.js)**

I decided to use solely and dimple.js as it would be sufficient for this task. I considered using multiple chart types (scatter, line chart, bubble chart, bar chart, etc.), color each line separately to test if this is a good way to visualize & stress on important point. I re-evaluated different chart type by tweaking few line of code and confirm my initial assumption, a stacked bar chart is already sufficient to dislay data characteristic. The first version is drawn from index-initial.html. This initial iteration can be viewed at index-initial.html, or below:
![First Chart](https://github.com/homagnibhatt/Udacity_Data_Analyst_Nanodegree_Projects/blob/master/Project_7_Make_Effective_Data_Visualization/initial_viz.PNG)

**Feedback**

I gathered feedback from 3 different people people and tried to follow Udacity questions guideline and here is the abridged responses.

**Interview #1**
> Your chart was a bit messy & no clear headlines & legend. If there is some explanation like a bold headline then these charts would make sense. The insights is not so much of a surprise and if you just create a small tweak in the legend, x-axis & y-axis then this would looks good. The data clearly favors your initial hypothesis, women children & elders are prioritized to board the baot first.

**Interview #2**
>The chart is interactive, that's nice. But why the second chart is revert and not in-line with the other chart. Switch x-axis & y-axis with each other and I think the chart will looks much better. By the way, the first chart to split between classes is cool but I think you can make it even better by combining gender & classes to see if there's different behavior in different classes. Broadly speaking, the chart looks intuitive & only needs a few small tweak.

**Interview #3**
>The second chart looks a bit weird and too much junk information, there's no need to include different age in different age bracket like that. There's not much information to show. And for the first chart, split the column into two, a stacked-bar wouldn't be necessary. Also, you also needs to clean up the headline & make clear of the axis, what is PClass? Can you makes it a bit clearer. Overall, this chart is straightforward.

**Post-feedback Design**

Following the feedback from the 3 interviews, I implemented the following changes:

I separate man & women from the first chart.
I added careful chart title & clearly labeled axis title.
I flipped the chart from horizontal bar chart to vertical bart chart.
I remove the individual age & only shows aggergrate age group.
I intended to add few special effect (highlight a chart when mouseover) but this would not be necessary.
I switched from Number of Survival to Survival Rates since the amount of passengers in each class/ages group is not similar.
Final rendition of the data visualization is shown below:

![Final Chart](https://raw.githubusercontent.com/tommyly2010/Udacity-Data-Analyst-Nanodegree/master/p6 - Data Visualization/img/image-final.png)

**Resources**

[dimple.js](http://dimplejs.org/)
[Data Visualization and D3.js (Udacity)](https://in.udacity.com/course/data-visualization-and-d3js--ud507)
[mbostock's blocks](https://bl.ocks.org/mbostock)

**Data**

train.csv: original downloaded dataset with minor cleaning for dimple.js implementation.
