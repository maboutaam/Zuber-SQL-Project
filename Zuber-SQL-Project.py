# Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import numpy as np
import datetime


# In[2]:


# Laoding CSV Files

df_company_trips = pd.read_csv("/datasets/project_sql_result_01.csv")


# ### The above dataframe is for the Company Trips data.

# In[3]:


# Loading CSV Files

df_dropoff_neighborhoods = pd.read_csv("/datasets/project_sql_result_04.csv")


# ### The above dataframe is for the Dropoff Neighborhoods data.

# In[4]:


# Load CSV Files

df_rides = pd.read_csv("/datasets/project_sql_result_07.csv")


# ### The above dataframe contains data on rides from the Loop to O'Hare International Airport.

# In[5]:


# working code

df_company_trips.info()


# In[6]:


# working code

df_dropoff_neighborhoods.info()


# In[7]:


# working code

df_rides.info()


# In[8]:


# working code

df_company_trips.dtypes


# In[9]:


# working code

df_dropoff_neighborhoods.dtypes


# In[10]:


# working code

df_rides.dtypes


# In[11]:


# working code

df_company_trips.describe()


# In[12]:


# working code

df_dropoff_neighborhoods.describe()


# In[13]:


# working code

df_rides.describe()


# In[14]:


# Prepare Data

# Convert missing values to NaN
df_company_trips = df_company_trips.replace("", np.nan)


# ### Missing values are converted to NaN to make it more convenient to analyze the data.
# 

# In[15]:


# Prepare Data

# Convert missing values to NaN
df_dropoff_neighborhoods = df_dropoff_neighborhoods.replace("", np.nan)


# ### Missing values are converted to NaN to make it more convenient to analyze the data.
# 

# In[16]:


# Prepare Data

# Convert missing values to NaN
df_rides = df_rides.replace("", np.nan)


# ### Missing values are converted to NaN to make it more convenient to analyze the data.
# 

# In[17]:


# Prepare Data

# Drop duplicate rows

df_company_trips = df_company_trips.drop_duplicates()


# ### Company Trips dataframe is clear from duplicates

# In[18]:


# Prepare Data

# Drop duplicate rows

df_dropoff_neighborhoods = df_dropoff_neighborhoods.drop_duplicates()


# ### Dropoff Neighborhoods dataframe is clear from duplicates.

# In[19]:


df_rides = df_rides.drop_duplicates()


# ### Rides dataframe is clear from duplicates

# In[20]:


df_company_trips.head()


# In[21]:


df_dropoff_neighborhoods.head()


# In[22]:


df_rides.head()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The data was loaded and inspected
#     
# </div>

# In[23]:


# Top 10 neighborhoods in terms of dropoffs

df_top_neighborhoods = df_dropoff_neighborhoods.sort_values(by='average_trips', ascending=False)

# Select the top 10 neighborhoods
top_10_neighborhoods = df_top_neighborhoods.head(10)

# Print
print(top_10_neighborhoods)


# ### Top 10 Neighborhoods in Terms of Drop-offs are: Loop, River North, Streeterville, West Loop, O'Hare, Lake View, Grant Park, Museum Campus, Gold Coast, Sheffield & DePaul.

# In[24]:


# Data Visualization

# Sort the data by number of rides in descending order
df_company_trips_sorted = df_company_trips.sort_values(by='trips_amount', ascending=False)

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='trips_amount', y='company_name', data=df_company_trips_sorted)
plt.title('Number of Rides by Taxi Company')
plt.xlabel('Number of Rides')
plt.ylabel('Taxi Company')
plt.show()


# ### The distribution of rides among the various taxi companies is shown in the graph. The taxi company with the highest number of rides is Flash Cab with approximately 18,000 ride and the lowest is RC Andrews Cab with 0 rides. This indicates that Flash Cab is the most popular taxi company in Chicago.

# In[25]:


# Data Visualization

# Sort the data by average number of trips in descending order
df_top_neighborhoods = df_dropoff_neighborhoods.sort_values(by='average_trips', ascending=False).head(10)

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='average_trips', y='dropoff_location_name', data=df_top_neighborhoods)
plt.title('Top 10 Neighborhoods by Number of Drop-offs')
plt.xlabel('Average Number of Drop-offs')
plt.ylabel('Neighborhood')
plt.show()


# ### The top 10 neighborhoods with the greatest average number of drop-offs can be seen on the graph. The first on this list Loop neighborhood with more than 10,000 dropoffs and the last is Sheffield & Paul with approximately 1600 dropoffs. This means that there is a high demand for taxi cab in the Loop neighborhood.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Top taxi companies and dropoff locations were identified successfully!
#     
# </div>

# In[26]:


# Testing Hypothesis

# Separate the data into two groups: rainy Saturdays and non-rainy Saturdays
rainy_saturdays = df_rides[df_rides['weather_conditions'] == 'Bad']['duration_seconds']
non_rainy_saturdays = df_rides[df_rides['weather_conditions'] == 'Good']['duration_seconds']


# In[27]:


# Testing Hypothesis

# Perform a two-sample t-test
t_statistic, p_value = ttest_ind(rainy_saturdays, non_rainy_saturdays)

# Significance level
alpha = 0.05

# Print the results
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Compare the p-value with the significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is sufficient evidence to conclude that the average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to conclude that the average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays.")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# The test was conducted correctly, conclusion is constistent with test results
#     
# </div>

# ### The hypothesis was tested to know wether the average duration of rides from the Loop neighborhood to O'Hare International Airport changes on rainy Saturdays.
# ### The data was separted into two groups: Rainy Saturdays and Non-rainy Saturdays.
# ### The separation is based on weather conditions to distinguished to distinguish a clearer data analysis.
# ### We chose the significance level alpha = 0.5 because it is the most popular method for testing hypothesis.
# ### The P-value comes out to be 7.397770692813658e-08 which is far less than 0.05 the valye of alpha.
# ### This means that we reject the null hypothesis, there is sufficient evidence to conclude that the average duration of rides from Loop neighborhoods to O'Hare International Airport changes on rainy Saturdays.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Makes sense!
#     
# </div>

# # Conclusion
# 
# ### Top 10 Neighborhoods in Terms of Drop-offs are: Loop, River North, Streeterville, West Loop, O'Hare, Lake View, Grant Park, Museum Campus, Gold Coast, Sheffield & DePaul.
# 
# ### The distribution of rides among the various taxi companies is shown in the graph. The taxi company with the highest number of rides is Flash Cab with approximately 18,000 ride and the lowest is RC Andrews Cab with 0 rides. This indicates that Flash Cab is the most popular taxi company in Chicago.
# 
# ### The top 10 neighborhoods with the greatest average number of drop-offs can be seen on the graph. The first on this list Loop neighborhood with more than 10,000 dropoffs and the last is Sheffield & Paul with approximately 1600 dropoffs. This means that there is a high demand for taxi cabs in the Loop neighborhood.
# 
# ### There is sufficient evidence to conclude that the average duration of rides from Loop neighborhoods to O'Hare International Airport changes on rainy Saturdays.
# 
# ### In order to beat the competition, Zuber needs to make sure the passengers are transported in less duration than other taxi companies.
# 
# ### Zuber should focues on the Top 10 neighboords in terms of Drop-offs to increase it's demand.
# 
# ### Zuber have to provide better customer experience to distinguish itself in the market.
# 
# ### Zuber's biggest competitor is Flash Cab, there is a real need to attract it's customers by building a loyalty program based on points for ride-shares, for example.
# 

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Nice summary!
#     
# </div>
