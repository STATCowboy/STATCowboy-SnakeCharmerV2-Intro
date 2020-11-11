#
# Author: Jamey Johnston
# Title: Code Like a Snake Charmer v2: Introduction to Python!
# Date: 2020/11/07
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/STATCowboy-SnakeCharmerV2-Intro 
#

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
import geopandas

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from settings import APP_DATA

# Function to categorize Credit Scores
def creditScoreClass(score):
    scr = (score[0])
    if scr <= 579:
        return "Very Poor"
    elif scr <= 669 and scr > 579:
        return "Fair"
    elif scr <= 739 and scr > 669:
        return "Good"
    elif scr <= 799 and scr > 739:
        return "Very Good"
    elif scr > 799:
        return "Exceptional"
    else:
        return "undef"

# Load Data into Pandas Dataframe
dfFannieMac = pd.read_excel(os.path.join(APP_DATA, "FreddieMacQ32007SF_Subset.xlsx")) 

# Look at the Data
dfFannieMac.head()
dfFannieMac.info()

# Graph Data
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

# Loan Defaults Plot
sb.countplot(x='LoanDefault',data=dfFannieMac, palette='hls')

# Drop Columns not going to use in Analysis
dfFannieMac.drop(['First Payment Date', 'First Time Homebuyer',
                  'Maturity Date', 'MSA', 'Original UPB', 'PPM Flag', 
                  'Product TYpe', 'Property Type', 'Postal Code', 
                  'Loan Sequence Number of historical_data1_Q32007', 
                  'Original Loan Term', 'Seller Name', 'Servicer Name', 
                  'Loan Sequence Number of Q32007', 'Zero Balance Code', 
                  'Y01', 'Validation', 'Training'],axis=1,inplace=True)
dfFannieMac.head()
dfFannieMac.info()

# Check for Nulls/ Missing Values
dfFannieMac.isnull().sum()

# Drop rows with missing data
dfFannieMac.dropna(inplace=True)
dfFannieMac.isnull().sum()

# Create Categorical Variable of Credit Score based on FICO
dfFannieMac['Credit Score Class'] = dfFannieMac[['Credit Score']].apply(creditScoreClass, axis=1)

# Credit Score Classification Defaults Plot
sb.countplot(x='Credit Score Class',data=dfFannieMac, palette='hls')

# Replace yes/no with 1/0 for Loan Default
dfFannieMac['LoanDefault'] = dfFannieMac['LoanDefault'].replace(["Yes", "No"], [1, 0])
dfFannieMac['LoanDefault']  = pd.to_numeric(dfFannieMac['LoanDefault'])


# Bring in States Shapefile to plot with GeoPandas
states = geopandas.read_file('data/usa-states-census-2014.shp')
states.head()
states.columns=['STATEFP', 'STATENS', 'AFFGEOID', 'GEOID', 'Property State', 'NAME', 'LSAD', 'ALAND', 'AWATER', 'region', 'geometry']
states.head()

# Create summary dataframe of Sum of Loan Defaults by State
states_default = dfFannieMac.groupby('Property State').agg({'LoanDefault': ['sum']}).reset_index()
states_default.columns=['Property State', 'Loan Default Count']

# Merge State Summary of Loan Default and State Shaepfile
states_default_shp = states.merge(states_default, on='Property State').dropna(axis=0).sort_values(by='Loan Default Count',ascending=False).reset_index().reset_index()
states_default_shp.head()

# Show Map with Count of Loan Defaults by State
fig = plt.figure(1, figsize=(25,15)) 
ax = fig.add_subplot()
states_default_shp.apply(lambda x: ax.annotate(s=x.NAME + "\n" + str(x['Loan Default Count']), xy=x.geometry.centroid.coords[0], ha='center', fontsize=10),axis=1)
states_default_shp.boundary.plot(ax=ax, color='Black', linewidth=.4)
states_default_shp.plot(column='Loan Default Count', ax=ax, figsize=(12, 12), cmap='coolwarm')
ax.text(-0.05, 0.5, 'Loan Defaults by State', transform=ax.transAxes,
        fontsize=20, color='gray', alpha=0.5,
        ha='center', va='center', rotation='90')

# Setup Dummy Variable for Occupancey Status
cred_score_class = pd.get_dummies(dfFannieMac['Credit Score Class'],drop_first=True)
cred_score_class.head()

# Setup Dummy Variable for Occupancey Status
occ_class = pd.get_dummies(dfFannieMac['Occupancy Status'],drop_first=True)
occ_class.head()

# Setup Dummy Variable for Channel
channel_class = pd.get_dummies(dfFannieMac['Channel'],drop_first=True)
channel_class.head()

# Setup Dummy Variable for Property State
prop_state_class = pd.get_dummies(dfFannieMac['Property State'],drop_first=True)
prop_state_class.head()

# Setup Dummy Variable for Loan Purpose
loan_purp_class = pd.get_dummies(dfFannieMac['Loan Purpose'],drop_first=True)
loan_purp_class.head()

# Take a look at it now!
dfFannieMac.head()

# Create a copy of the Loan Default to move to position 0
loan_def = dfFannieMac['LoanDefault'] 

# Drop all the columsn we converted to Dummy Columns and Loan Default
dfFannieMac.drop(['LoanDefault', 'Credit Score', 'Credit Score Class', 'Occupancy Status', 'Channel', 
                  'Property State', 'Loan Purpose'],axis=1,inplace=True)
dfFannieMac.head()

dfFannieMac = pd.concat([loan_def, dfFannieMac, cred_score_class, occ_class, channel_class, 
                        prop_state_class, loan_purp_class],axis=1)
dfFannieMac.head()

# Correlation plots

dfFannieMacContVars = dfFannieMac.iloc[:,0:6]

corr = dfFannieMacContVars.corr()
sb.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# sb.pairplot(dfFannieMacContVars)

# g = sb.PairGrid(dfFannieMacContVars, diag_sharey=False)
# g.map_lower(sb.kdeplot, cmap="Blues_d")
# g.map_upper(plt.scatter)
# g.map_diag(sb.kdeplot, lw=3)

# Get Independent (X) and Dependent (y) variables for Log Reg Model
dfFannieMac.info()
X = dfFannieMac.iloc[:,1:70].values
y = dfFannieMac.iloc[:,0].values

# Way to use names of columns
# X = dfFannieMac.loc[:,['Mortgage Insurance%', 'Number of Units', 'Original CLTV']].values
# y = dfFannieMac.loc[:,'LoanDefault'].values

# Split Train and Test Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

# Run Logistic Regression using SciKit-Learn
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

# Use Test dataset to validate against Training
y_pred = LogReg.predict(X_test)

# Show a confusion Matrix to see how we did!
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

print(classification_report(y_test, y_pred))

# Plot a confusion Matrix to see how we did!
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')

print('coef:', LogReg.coef_, end='\n\n')