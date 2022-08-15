# Importing libraries
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from helpers.eda import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Task 1: Import the titanic dataset from the seaborn library
df = sns.load_dataset('titanic')
df.columns = [col.lower() for col in df.columns]
df.head()

# Task 2: Find the number of female and male passengers
df['sex'].value_counts()

# Task 3: Find the number of unique values belong to each column
df.nunique()

# Task 4: Find the unique values of the variable pclass
df['pclass'].unique()
df.loc[:, 'pclass'].unique()
df.pclass.head()

# Task 5: Find the number of the unique values of the variables pclass and parch
df.loc[:, ['pclass', 'parch']].nunique()
df[['pclass', 'parch']].nunique()

# Task 6: Ckeck the type of the variable embarked. Change its type to category and check its type again.
df['embarked'].dtype
df['embarked'] = df['embarked'].astype('category')
df['embarked'].dtype
str(df['embarked'].dtype)

# Task 7: Show all information for those with an embarked value of C
df[df['embarked'] == 'C'].shape
df[(df['embarked'] == 'C') & (df['sex'] == 'male')].shape

# Task 8: Show all information for those with no embarked value S
df[df['embarked'] != 'S'].head()
df[~(df['embarked'] == 'S')].head()

# Task 9: Take the number of female passengers older than 30 years old
df[(df['sex'] == 'female') & (df['age'] > 30)].head()

# Task 10: Show information for passengers whose Fare is over 500 or 70 years of age
df[(df['age'] > 70) | (df['fare'] > 500)].head()

# Task 11: Show the number of the null values in each variable
df.isnull().sum().sort_values(ascending=False)

# Task 12: Drop the variable 'who' from the dataframe
df.drop('who', axis=1, inplace=True)
df.head()

# Task 13: Fill the null values in the variable 'deck' with its mode
df['deck'].fillna(df['deck'].mode()[0], inplace=True)
df['deck'].isnull().sum()

# Task 14: Fill the null values in the variable 'age' with its median
df['age'].fillna(df['age'].median(), inplace=True)
df.age.isnull().sum()

# Task 15: Find the sum, count, mean values of the pclass and sex variables of the survived variable
df.groupby(['pclass', 'sex']).agg({'survived': ['sum', 'count', 'mean']})


# Task 16: Write a function that returns 1 for those under 30 and 0 for those above or equal to 30
# Create a variable named age_flag in the titanic data set using the function you wrote
def age_30(age):
    if age < 30:
        return 1
    else:
        return 0


df['age_flag'] = df['age'].apply(lambda x: age_30(x))

df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)
df.head()

# Task 17: Define the Tips dataset from the Seaborn library.
df = sns.load_dataset('tips')
df.columns = [col.lower() for col in df.columns]
df.head()

# Task 18: Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the time variable
df.groupby('time').agg({'total_bill': ['sum', 'min', 'max']})

# Task 19: Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the time and day variables
df.groupby(['time', 'day']).agg({'total_bill': ['sum', 'min', 'max']})

# Task 20: Find the sum, min, max and average of the total_bill and type values of the lunch time and female customers according to the day
df.head()
df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')].groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean']})

# Task 21: Find the average of orders with size less than 3 and total_bill greater than 10
df.loc[((df['size'] < 3) & (df['total_bill'] > 10)), 'total_bill'].mean()

# Task 22: Create a new variable called total_bill_tip_sum. Let him give the sum of the total bill and tip paid by each customer.
df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
df.head()

# Task 23: Find the mean of the total_bill variable for men and women separately.
# Create a new total_bill_flag variable that gives 0 for those below the averages you found, and one for those above and equal.
# Attention !! For females, the average for females will be taken into account, and for male, the average for males will be taken into account.
# start by writing a function that takes a gender and total_bill as a parameter. (will include if-else conditions)

mean_female = df[df['sex'] == 'Female']['total_bill'].mean()
mean_male = df[df['sex'] == 'Male']['total_bill'].mean()


def function(sex, total_bill):
    if sex == 'Female':
        if total_bill < mean_female:
            return 0
        else:
            return 1
    else:
        if total_bill < mean_female:
            return 0
        else:
            return 1


df['total_bill_flag'] = df[['sex', 'total_bill']].apply(lambda x: function(x['sex'], x['total_bill']), axis=1)
df.head()

# Task 24: Observe the number of below and above average by sex using the total_bill_flag variable
df.groupby(['sex', 'total_bill_flag']).agg({'total_bill_flag': 'count'})

# Task 25: Sort from largest to smallest according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe
new_df = df.sort_values('total_bill_tip_sum', ascending=False)[0:30]