
# coding: utf-8

# Intro to Data Analysis Project (Titanic Dataset)
# 
# Introduction
# Performing a data analysis on a sample Titanic dataset.
# This dataset contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic. A description of this dataset can be viewed here (https://www.kaggle.com/c/titanic/data).
# 
# 
# 
# Questions to be answered
# 
# The fundamental question that requires to be answered is provided very succicently at https://www.kaggle.com/c/titanic which is quite simply:- What factors made people more likely to survive?
# 
# There are nuances within this fundamental question which promises to give out interesting observations and conclusions. Hence, the following questions were chosen:- 
# 1.	Is there a relationship between socio economic status and survival?
# 2.	Is there a relationship between ticket price and survival?
# 3.	Did women and children have better survival?
# 4.	Did the presence of parents help survival of children compared to nannies?
# 5.	Did port of embarkation have any relation to survival, regardless of sex?
# 
# Assuming that everyone who survived made it to a life boat and it wasn't by luck.
# 
# 
# Data Wrangling
# 
# The data was obtained from https://www.kaggle.com/c/titanic  and it has the following characterictics:
# 
# •	survival: Survival (0 = No; 1 = Yes)
# 
# •	pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 
# •	name: Name
# 
# •	sex: Sex
# 
# •	age: Age
# 
# •	sibsp: Number of Siblings/Spouses Aboard
# 
# •	parch: Number of Parents/Children Aboard
# 
# •	ticket: Ticket Number
# 
# •	fare: Passenger Fare
# 
# •	cabin: Cabin
# 
# •	embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# 
# Special Notes:
# 
# •	Pclass is a proxy for socio-economic status (SES) 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# •	Age is in Years; Fractional if Age less than One (1) If the Age is Estimated, it is in the form xx.5
# 
# Some relations were ignored from the variables sibsp and parch. The following are the relations considered. The relations ignored are cousins, nephew/niece, aunts/uncle and in-laws. Children travelling with nanny have parch=0 and hence they were ignored. People travelling with neighbors/friends were also ignored. This was done to render the dataset compact and limited to immediate family so that a sharper analysis can be done.
# 
# Hence, following are the relationships considered finally:
# 
# •	Sibling: Brother, Sister, Stepbrother, or Stepsister of Passenger aboard Titanic only
# 
# •	Spouse: Husband or Wife of Passenger aboard Titanic only.
# 
# •	Parent: Mother or Father of Passenger aboard Titanic only.
# 
# •	Child: Son, Daughter, Stepson, or Stepdaughter of Passenger aboard Titanic only.
# 
# 

# In[32]:


# Render plots inline
get_ipython().magic('matplotlib inline')

# Import various libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


# Set style for graphs
sns.set_style("dark")


# In[34]:


# Create dataframe
titanic_data = pd.read_csv('titanic-data.csv')


# In[35]:


# Create Survival Label Column
titanic_data['Survival'] = titanic_data.Survived.map({0 : 'Died', 1 : 'Survived'})
titanic_data.Survival.head()

# Create Pclass Label Column
titanic_data['Class'] = titanic_data.Pclass.map({1 : 'First Class', 2 : 'Second Class', 3 : 'Third Class'})
titanic_data.Class.head()

# Create Embarked Labels Column
titanic_data['Ports'] = titanic_data.Embarked.map({'C' : 'Cherbourg', 'Q' : 'Queenstown', 'S' : 'Southampton'})

# Create Survival_Rate Label Column as Survived Copy
titanic_data['Survival_Rate'] = titanic_data.Survived


# In[36]:


# Print the first few records to review data and format
titanic_data.head()


# In[37]:


# Print the last few records to review data and format
titanic_data.tail()


# Data Cleanup
# From the data description and questions to answer, it can be safely surmised that some dataset columns will not play a part in the analysis and these columns can therefore be removed. This will de-cluster the dataset and help with improve processing performance of the dataset.
# •	PassengerId
# •	Name
# •	Ticket
#  The approach to data cleanup has been taken as follows:
# 1.	Identifying and removing any duplicate entries
# 2.	Removing unnecessary columns
# 3.	Fixing missing and data format issues
# 
# Step 1 - Remove duplicate entries
# No duplicate entries exist, based on the following tests below.
# 

# In[38]:


# Identify and remove duplicate entries
titanic_data_duplicates = titanic_data.duplicated()
print ('Number of duplicate entries is/are {}'.format(titanic_data_duplicates.sum()))


# In[39]:


# Duplicate test
duplicate_test = titanic_data.duplicated('Age').head()
print ('Number of entries with duplicate age in top entires are {}'.format(duplicate_test.sum()))
titanic_data.head()


# Step 2 - Remove unnecessary columns
# Columns (PassengerId, Name, Ticket) removed

# In[40]:


# Create new dataset without PassengerID, Name, Ticket, Cabin
titanic_data_cleaned = titanic_data.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
titanic_data_cleaned.head()


# Step 3 - Fix any missing or data format issues

# In[41]:


# Find number of missing values
titanic_data_cleaned.isnull().sum()


# In[42]:


# Review some of the missing Age data
missing_age_bool = pd.isnull(titanic_data_cleaned['Age'])
titanic_data_cleaned[missing_age_bool].head()


# In[43]:


# Determine number of males and females with missing age values
missing_age_female = titanic_data_cleaned[missing_age_bool]['Sex'] == 'female'
missing_age_male = titanic_data_cleaned[missing_age_bool]['Sex'] == 'male'


# In[44]:


print ('Number for females and males with age missing are {} and {} respectively'.
       format(missing_age_female.sum(),missing_age_male.sum()))


# In[45]:


# Taking a look at the datatypes
titanic_data_cleaned.info()


# Missing Age data is 177 out of 891 i.e about 20% of our dataset. Graphing and summations shouldn't be a problem since they will be treated as zero(0) value. However,it needs to be accounted for if reviewing descriptive stats such as mean age.
# 
# Age missing proportions across male and female are
# 
# •	Age missing in male data: 124
# 
# •	Age missing in female data: 53
# 

# Data Exploration and Visualization

# In[46]:


# Descriptive statistics for cleaned dataset
titanic_data_cleaned.describe()


# Question 1
# 
# Were social-economic standing a factor in survival rate?
# 

# In[47]:


# Survival rate/percentage of sex and class
def survival_rate(Class, sex):
    """
    Args:
        Class: class value First Class,Second Class or Third Class
        sex: male or female
    Returns:
        survival rate as percentage.
    """
    grouped_by_total = titanic_data_cleaned.groupby(['Class', 'Sex']).size()[Class,sex].astype('float')
    grouped_by_survived_sex =         titanic_data_cleaned.groupby(['Class','Survived','Sex']).size()[Class,1,sex].astype('float')
    survived_sex_pct = (grouped_by_survived_sex / grouped_by_total * 100).round(2)
    
    return survived_sex_pct



# In[48]:


# Actual numbers grouped by class, survival and sex
groupedby_class_survived_size = titanic_data_cleaned.groupby(['Class','Survived','Sex']).size()


# In[49]:


# Graph - Grouped by class, survival and sex
g = sns.factorplot(x="Sex", y="Survival_Rate", col="Class", 
                   data=titanic_data_cleaned,
                    kind="bar", ci=None, size=5, aspect=.8)


# In[50]:


# Graph - Grouped by class, survival and sex
g = sns.factorplot(x="Sex", y="Survival_Rate", col="Class", data=titanic_data_cleaned,
                   saturation=.5, kind="bar", ci=None, size=5, aspect=.8)


# In[51]:


# Grouped by class, survival and sex
print (groupedby_class_survived_size)
print ('First Class - female survival rate: {}%'.format(survival_rate('First Class','female')))
print ('First Class - female survival rate: {}%'.format(survival_rate('First Class','male')))
print ('-----')
print ('Second Class - female survival rate: {}%'.format(survival_rate('Second Class','female')))
print ('Second Class - female survival rate: {}%'.format(survival_rate('Second Class','male')))
print ('-----')
print ('Third Class - female survival rate: {}%'.format(survival_rate('Third Class','female')))
print ('Third Class - female survival rate: {}%'.format(survival_rate('Third Class','male')))


# Looking at the percentages of the overall passengers per class and the total numbers across each class, it can be assumed that a passenger from Class 1 is about 2.5x times more likely to survive than a passenger in Class 3.
# 
# 
# Social-economic standing was a factor in survival rate of passengers.
# 
# •	First Class : 62.96%
# 
# •	Second Class: 47.28%
# 
# •	Third Class : 24.24%
# 

# 
# Question 2
# 
# Is there a relationship between ticket price and survival?
# 
# So, to determine this relation and properly classify this, we first quantify the fare data into 3 bands- high, medium and low, which reflects the relative wealth of the passengers. This is done by first determining the quartiles. This will divide the the fares into 3 wealth levels as
# 
# 1.	1st Quartile (<25) = Low
# 2.	2nd, 3rd Quartile (25 to 75)= Medium
# 3.	4th Quartile (>75) = High
# 

# In[54]:


fare_quantiles= pd.qcut(titanic_data_cleaned['Fare'], 
                        q= [0, 0.25, 0.5, 1], 
                        labels= ['Low', 'Medium', 'High'])

gb= titanic_data_cleaned.groupby(fare_quantiles)['Survived'].mean()*100
ax=gb.plot.bar()

ax.set(title='Survival rate according to Fare', 
       xlabel= 'Wealth', ylabel=' Survival Rate as %',
       ylim=[0,100])

# Fix rotation on y-ticks
ax.set_xticklabels(ax.get_xmajorticklabels(), rotation=0)


   


# From the percentages, it is obvious that passengers with higher Wealth i.e. passengers booking higher ticket value was a factor in survival.

# 
# Question 3
# 
# Did women and children have better survival rate?
# 
# Since children are not defined in this dataset, hence passengers with age<18 is taken to be children.
# 
# 

# In[29]:


# Create Cateogry column and categorize people

titanic_data_cleaned.loc[
    ( (titanic_data_cleaned['Sex'] == 'female') & 
    (titanic_data_cleaned['Age'] >= 18) ),'Category'] = 'Woman'


# In[30]:


titanic_data_cleaned.loc[
    ( (titanic_data_cleaned['Sex'] == 'male') & 
    (titanic_data_cleaned['Age'] >= 18) ),
    'Category'] = 'Man'


# In[31]:


titanic_data_cleaned.loc[
    (titanic_data_cleaned['Age'] < 18),
    'Category'] = 'Child'


# In[32]:


# Graph - Compare survival count between Men, Women and Children
g = sns.factorplot(x='Survival', col='Category', data=titanic_data_cleaned, kind='count', size=7, aspect=.8)


# In[44]:


# Get the totals grouped by Men, Women and Children, and by survival
print (titanic_data_cleaned.groupby(['Category','Survived']).size())


# The data and the graphs suggests that being "women and children" possibly played a role in the survival of a number of people.
# It seems from the data that more women survived than children percent wise which is a bit surprising. It is possible that near adults ie close to 18 children particularly boys were treated as adults and hence were left to fend for themselves as adults, hence accounting for the larger mortality.
# 

# Question 4
# 
# Did the presence of parents help survival of children compared to nannies?
# 
# Some of the children were accompanied by their nannies. Now an interesting question is whether they abandoned their wards to save themselves or took similar care as any parent would and try to save the children. So, first we have to separate the children into two groups, one with nanies and other with parents and compare their survival rates.
# 
# 
# Assumptions:
# 
# Classifying people as 'Child' represented by those under 18 years old is applying today's standards to the 1900 century
# 

# In[45]:


# Parse out children with parents from those with nannies 
titanic_data_children_nannies = titanic_data_cleaned.loc[
    (titanic_data_cleaned['Category'] == 'Child') &
    (titanic_data_cleaned['Parch'] == 0)]


# In[46]:


titanic_data_children_parents = titanic_data_cleaned.loc[
    (titanic_data_cleaned['Category'] == 'Child') &
    (titanic_data_cleaned['Parch'] > 0)]


# In[47]:


# Determine children with nannies who survived and who did not
survived_children_nannies = titanic_data_children_nannies.Survived.sum()
total_children_nannies = titanic_data_children_nannies.Survived.count()
pct_survived_nannies = ((float(survived_children_nannies)/total_children_nannies)*100)
pct_survived_nannies = np.round(pct_survived_nannies,2)
survived_children_nannies_avg_age = np.round(titanic_data_children_nannies.Age.mean())


# In[48]:


# Results
print ('Total number of children with nannies: {}\nChildren with nannies who survived: {}\nChildren with nannies who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'.format(total_children_nannies, survived_children_nannies, 
        total_children_nannies-survived_children_nannies, pct_survived_nannies, survived_children_nannies_avg_age))


# In[49]:


# Determine children with parents who survived and who did not
survived_children_parents = titanic_data_children_parents.Survived.sum()
total_children_parents = titanic_data_children_parents.Survived.count()
pct_survived_parents = ((float(survived_children_parents)/total_children_parents)*100)
pct_survived_parents = np.round(pct_survived_parents,2)
survived_children_parents_avg_age = np.round(titanic_data_children_parents.Age.mean())


# In[50]:


# Results
print ('Total number of children with parents: {}\nChildren with parents who survived: {}\nChildren with parents who did not survive: {}\nPercentage of children who survived: {}%\nAverage age of surviving children: {}'.format(total_children_parents, survived_children_parents, 
        total_children_parents-survived_children_parents, pct_survived_parents,survived_children_parents_avg_age))


# Based on the data analysis above, it would appear that the survival rate for children who were accompanied by parents vs those children accompanied by nannies was slighly higher for those with parents. 
# 
# The slight increase could be due to the average age of children with parents being younger, almost half, that of children with nannies.
# 
# •	Percentage of children with nannies who survived: 50.0%
# 
# •	Percentage of children with parents who survived: 55.56%
# 
# •	Average age of surviving children with nannies: 15
# 
# •	Average age of surviving children with parents: 7.0
# 

# Question 5
# 
# Did port of embarkation have any relation to survival, regardless of sex?
# 
# We first determine the number of people embarking from each port and plot it is follows

# In[56]:


titanic_data_cleaned.Ports.value_counts().plot(kind='pie')
plt.axis('equal')
plt.title('Number of appearances in dataset')


# Thus Southamption has the maximum number of people embarking. However, what we really need to find is the percentage of survivors from each port which will give us a better picture.

# In[51]:


# Define Survival and Embarked
Embarked= titanic_data_cleaned['Embarked']
Survived= titanic_data_cleaned['Survived']


# In[201]:


# Survival rate/percentage of Embarked ports
def survival_rate(Embarked):
    """
    Args:
        Embarked: class value S,C or Q
    Returns:
        survival rate as percentage.
    """
    grouped_by_total = titanic_data_cleaned.groupby(['Embarked']).size()[Embarked].astype('float')
    grouped_by_survived_Embarked =         titanic_data_cleaned.groupby(['Embarked','Survived']).size()[Embarked,1].astype('float')
    survived_Embarked_pct = (grouped_by_survived_Embarked / grouped_by_total * 100).round(2)
    
    return survived_Embarked_pct
   


# In[203]:


# Actual numbers grouped by suvival and Embarked
groupedby_Embarked_survived = titanic_data_cleaned.groupby(['Embarked','Survived']).size()


# In[204]:


# Grouped by Survival and Embarked

print (groupedby_Embarked_survived)
print ('Southampton - survival rate: {}%'.format(survival_rate('S')))
print ('-----')
print ('Cherbourg - survival rate: {}%'.format(survival_rate('C')))
print ('-----')
print ('Queenstown - survival rate: {}%'.format(survival_rate('Q')))


# Hence, Survival rate of passengers from Cherbourg was the highest and hence port of embarkation was a factor in survival.

# Conclusion
# 
# The results of the analysis, although tentative, indicates that wealth, class and sex, namely, being a wealthy female with upper social-economic standing (first class), would give one the best chance of survival when the tragedy occurred on the Titanic. While being a man in third class, gave one the lowest chance of survival. Women and children, across all classes, tend to have a higher survival rate than men in genernal but by no means did being a child or woman guarentee survival. 
# Although, overall, children accompanied by parents (or nannies) had the best survival rate at over 50%.
# 
# Issues:
# •	A portion of men and women did not have Age data and were removed from calculations which could have skewed some numbers
# 
# •	The category of 'children' was assumed to be anyone under the age of 18, using today's North American standard for adulthood which was certainly not the case in the 1900s
# 
# References
# •	https://www.kaggle.com/c/titanic/data
# 
# •	http://nbviewer.jupyter.org/github/jvns/pandas-cookbook/tree/master/cookbook/
# 
# •	https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.factorplot.html#seaborn.factorplot
# 
# •	http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3865739/
# 
