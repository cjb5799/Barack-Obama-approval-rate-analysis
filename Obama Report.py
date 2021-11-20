#!/usr/bin/env python
# coding: utf-8

# In[224]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[225]:


#Step 1.Load the data from the “obama-approval-ratings.xls” file into a DataFrame.
dirname = '/Users/carla/Desktop/Summer 2021/DSC640/Week 1/obama_approval_rating.csv'
df = pd.read_csv(dirname)



# In[165]:


missing_values_count_df= df.isnull().sum()
missing_values_count_df[0:14]


# In[166]:


#- Display the type of variables of dataframe
df.head()
df.dtypes


# In[167]:


print("The dimension of the table 1 is: ",  df.shape)


# In[168]:


#Display bar chart,

import matplotlib.pyplot as plt

approve=df['Approve']
disapprove=df['Disapprove']

plt.bar(approve, disapprove)
plt.title('Approval Vs Disapproval')
plt.xlabel('Approval')
plt.ylabel('Disapproval')
plt.show()


# In[169]:


#1 Display stacked bar chart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
df_grouped = df.groupby('Issue').sum()
[['Approve','Disapprove','None']]
df_grouped
# define figure
fig, ax = plt.subplots(1, figsize=(16, 6))
# numerical x
x = np.arange(0, len(df_grouped.index))
# plot bars
#plt.bar(x - 0.3, df_grouped['Issue'], width = 0.2, color = '#1D2F6F')
plt.bar(x - 0.1, df_grouped['Approve'], width = 0.2, color = '#8390FA')
plt.bar(x + 0.1, df_grouped['Disapprove'], width = 0.2, color = '#6EAF46')
plt.bar(x + 0.3, df_grouped['None'], width = 0.2, color = '#FAC748')

# remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# x y details
plt.ylabel('Rate Score')
plt.xticks(x, df_grouped.index)
plt.xlim(-0.5, 31)
# grid lines
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
# title and legend
plt.title('Obama Approval and Disapproval Rate Score', loc ='left')
plt.legend(['Approve', 'Disapprove', 'None'], loc='upper left', ncol = 4)
plt.show()


# In[220]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.graph_objs as go


exp_vals =df['Approve'] 
exp_labels = df['Issue']
#plt.pie(exp_vals,labels=exp_labels)

plt.pie(exp_vals,labels=exp_labels, shadow=True, autopct='%1.1f%%',radius=1.5)
plt.show()



# In[221]:


#1 donut, 
  
import matplotlib.pyplot as plt
share =df['Disapprove'] 
labels = df['Issue']
autopct=['%.2f%%']
plt.style.use('ggplot')
plt.title('Obama  Disapproval Rate Score')
plt.pie(x=share, labels=labels, autopct='%.2f%%',startangle=100)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#plt.legend(loc='upper right')

my_circle=plt.Circle( (0,0), 0.9, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.show()


# In[222]:


#1 line chart with Python
df.columns = ['Issue','Approve','Disapprove','None']
df1 = df['Issue']
df =df1
layout = dict(title='Chart from Pandas DataFrame', xaxis= dict(title='x-axis'), yaxis= dict(title='y-axis'))

#df.iplot(filename='cf-simple-line-chart', layout=layout)


# In[228]:


import matplotlib.pyplot as plt

df_approve = df['Approve']
df_Disapprove = df['Disapprove']

#Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
#Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
  
plt.plot(df_approve, df_Disapprove)
plt.title('Disapprove Rate Vs Approve Rate')
plt.xlabel('Approval Rate')
plt.ylabel('Disapproval Rate')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




