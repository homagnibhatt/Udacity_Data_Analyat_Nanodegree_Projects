**Data Visualization: Titanic Data Visualization** 

by Homagni Bhattacharjee, as Project 7 of Udacity's Data Analyst Nanodegree.

**Summary**

This project attempts to understand the age, sex and economic profile of the people who survived the Titanic disaster. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, upper-class etc as they may first access to lifeboats. The following chart attempts to bring out these factors via a storyboard bar chart. It shows the following:

1.  The animating feature is the Age_Bucket (creating by grouping the age into defined groups in the R script).
2.  The bar chart gives a nice overview of the impact of Class, Sex, Count and Age factors in the survival of the Titanic passengers.

**Design**

**Exploratory Data Analysis and Cleaning (R)**

I downloaded the data from [Kaggle - Machine Learning from Disaster](https://www.kaggle.com/c/titanic), selecting a [train data set](https://www.kaggle.com/c/titanic/data) which already had basic finding & doesn't need extensive data wrangling or transformation.

Although the dataset is fairly cleaned, some preliminary cleaning was performed using R. The following steps were taken:
1.	Removing rows with NA and blank values so that a complete dataset was used.
2.	Since we are interested in looking into the survivors and the factors which influence it, we will only use data where Survived = 1 i.e. only those data points where the passenger survived.
3.	Here, we make a preliminary observation in the dataset. Hence, we will look into only a limited number of variable and its interplay with survival.
4.	For this reason we only use the following variables: Survived, PClass, Sex, Age, Embarked.

**Data Visualization (dimple.js)**

I decided to use solely and dimple.js as it would be sufficient for this task. I considered using multiple chart types (scatter, line chart, bubble chart, bar chart, etc.), to test if this is a good way to visualize & stress on important point. I re-evaluated different chart type by tweaking few line of code and confirm my initial assumption, a bar chart is already sufficient to dislay the required data characteristic. The first version is drawn from index-initial.html. This initial iteration can be viewed at [index-initial.html](https://github.com/homagnibhatt/Udacity_Data_Analyst_Nanodegree_Projects/blob/master/Project_7_Make_Effective_Data_Visualization/index_initial.html), or below:

![First Chart](https://github.com/homagnibhatt/Udacity_Data_Analyst_Nanodegree_Projects/blob/master/Project_7_Make_Effective_Data_Visualization/image_initial.PNG)

**Feedback**

I gathered 3 feedbacks and tried to follow Udacity questions guideline. Here is the abridged responses.

**Feedback #1**
> Your chart was a bit messy & no clear headlines & legend. If there is some explanation like a bold headline then these charts would make sense. The data clearly favors your initial hypothesis, women, & upper class are prioritized to board the baot first.

  > Title and explanatory text is introduced. However, separate legend is not used as the current labels are adequate.

**Feedback #2**
>The visualization to be explanatory rather than exploratory. To achieve this, I suggest including a paragraph describing the finding(s) that you wish readers to know from reading the visualization..
  
  > Explanatory text is used..
  
**Feedback #3**
> I also think that there are too many variables included in the plot that it is hard to appreciate the most prominent trend in the plot. Reduce the number of variables or update the plot to ensure the main finding(s) is easily visible in the visualization.

  >  Variables are reduced and ony, PClass, Sex and Age is used.
  
**Conclusion**

The feedback was incorporated and the final chart was drawn at [index_final.html](https://github.com/homagnibhatt/Udacity_Data_Analyst_Nanodegree_Projects/blob/master/Project_7_Make_Effective_Data_Visualization/index_final.html). The general assumption of more women survivors was found justified. Upper class passengers had more survivors overall but curiously, the number of survivors in each class seemed to be inveresly propertional to age. This could be due to the fact that upper class passengers had fewer children than lower class  passengers which shows in the figures. However, tis needs furthur digging.

![Final Chart](https://github.com/homagnibhatt/Udacity_Data_Analyst_Nanodegree_Projects/blob/master/Project_7_Make_Effective_Data_Visualization/image_final.gif)

**Resources**

[dimple.js](http://dimplejs.org/)

[Data Visualization and D3.js (Udacity)](https://in.udacity.com/course/data-visualization-and-d3js--ud507)

[mbostock's blocks](https://bl.ocks.org/mbostock)
