# CreditCard-Customers-Classification
### The objective here was to create a Machine Learning Model able to identify credit card clients who spend low periods on books in order to be able to directly adress them and try to convince them to stay longer.

First we will understand our dataset with a **Exploratory Analysis**<br>
<br>
The second step is a **Feature Engeniring** to identify the relevant features to our classification model and treat them to fit our model design<br>
<br>
The third and final step is **Model Desing**, where we'll apply three models and compare their results<br>
<br>

Let's begin!!
<br>

``` ruby
#### Import libraries #### 

## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import missingno as msno

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, svm
import xgboost
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
```

### Exploratory data analysis

``` ruby
dtf = pd.read_csv("BankChurners.csv")
display(dtf)
```

this is how the initial dataset looks like

Output: 
<p align="center">
<img src=Images-plots/initial_dataset.png />
</p>

Dtypes and Missing data

``` ruby
# create function to evaluate how many categorical and numerical variables there are and the proportion of missing data
def utils_recognize_type(dtf, col, max_cat = 20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() <= max_cat):
        return "cat"
    elif dtf[col].dtype == 'datetime64[ns]':
        return "datetime"
    elif dtf[col].dtype == 'bool':
        return "bool"
    else:             
        return "num"
```
``` ruby
dic_cols = {col:utils_recognize_type(dtf, col, max_cat=5) for col in dtf.columns}
print(dic_cols)
```
``` ruby
# create heatmap to visualize features types in dataset
heatmap = dtf.isnull()

for k,v in dic_cols.items():
    #if v == "datetime":
        #heatmap[k] = heatmap[k].apply(lambda x: 2 if x is False else 1)
    if v == "num":
        heatmap[k] = heatmap[k].apply(lambda x: 0)
    elif v == "bool":
        heatmap[k] = heatmap[k].apply(lambda x: 0.5)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 1)
        

fig, ax = plt.subplots(figsize=(18,7))
sns.set_style({'axes.labelcolor': 'white'})
sns.heatmap(heatmap, cbar = False).set_title("Dataset Overview")
plt.show()

print("\033[1;37;40m Numeric ", "\033[1;30;47m Categorical ", "\033[1;30;41m Boolean ")
```
Output: 
<p align="center">
<img src=Images-plots/data_types_plot.png />
</p>

``` ruby
# Visualize missing values with missingno
msno.matrix(dtf)
```
Output: 
<p align="center">
<img src=Images-plots/missing_data_plot.png />
</p>

We have no missing data. GREAT!

``` ruby
# Set the "Id" column as Index and rename "Months_on_book" as "Y"
dtf = dtf.set_index("CLIENTNUM")

dtf = dtf[['Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count',
       'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Months_on_book']]

dtf = dtf.rename(columns = {"Months_on_book":"Y"})
```

Numerical variables analysis

``` ruby
def int_feature(x = dtf.columns[len(dtf.columns)-1]):
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize = (15,5))
    fig.suptitle(x, fontsize=20)
    ### distribution
    ax[0].title.set_text('distribution')
    variable = dtf[x].fillna(dtf[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                        breaks[10]) ]
    
    sns.histplot(variable, kde = True, fill = True, ax=ax[0], element = "step")
    des = dtf[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ### boxplot 
    ax[1].title.set_text('outliers (log scale)')
    tmp_dtf = pd.DataFrame(dtf[x])
    tmp_dtf[x] = np.log(tmp_dtf[x])
    tmp_dtf.boxplot(column=x, ax=ax[1])
    plt.show()  
```
``` ruby
num_cols = [col for col in dtf.columns if utils_recognize_type(dtf, col = col, max_cat=20) == "num"]
cat_cols = [col for col in dtf.columns if utils_recognize_type(dtf, col = col, max_cat=20) == "cat"]

widgets.interact(int_feature, x = dtf[num_cols])
```
Output: (Result Exemples)
<p align="center">
<img src=Images-plots/NA_Y_plot.png />
</p>
<p align="center">
<img src=Images-plots/NA_Customer_Age.png />
</p>
<p align="center">
<img src=Images-plots/NA_Total_Trans_Amt.png />
</p>
<p align="center">
<img src=Images-plots/NA_Total_Trans_Ct.png />
</p>

Categorical variables analysis

``` ruby
# Plot a bar plot to understand labels frequency for a single categorical variabel
def cat_feature(x = dtf.columns[len(dtf.columns)-1], y = dtf.columns[len(dtf.columns)-2]):
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize = (25,5))
    fig.suptitle("Bar Plots", fontsize=20)
    
    ax[0].title.set_text(x)
    ax[0] = dtf[x].value_counts().sort_values().plot(kind="barh", ax = ax[0])
    totals= []
    for i in ax[0].patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax[0].patches:
         ax[0].text(i.get_width()+.3, i.get_y()+.20, 
         str(round((i.get_width()/total)*100, 2))+'%', 
         fontsize=10, color='black')
    ax[0].grid(axis="x")    
    
    ax[1].title.set_text(y)
    ax[1] = dtf[y].value_counts().sort_values().plot(kind="barh", ax = ax[1])
    totals= []
    for i in ax[1].patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax[1].patches:
         ax[1].text(i.get_width()+.3, i.get_y()+.20, 
         str(round((i.get_width()/total)*100, 2))+'%', 
         fontsize=10, color='black')
    ax[1].grid(axis="x")    
    plt.show()
```
``` ruby
widgets.interact(cat_feature, x = dtf[cat_cols], y = dtf[cat_cols])
```
Output: (Result Exemples)
<p align="center">
<img src=Images-plots/CA_GenderXEducation_Level.png />
</p>
<p align="center">
<img src=Images-plots/CA_Marital_StatusXTotal_Relationship_Count.png />
</p>

Categorical VS Y

``` ruby
def cat_vs_Y(cat = dtf.columns[len(dtf.columns)-2]):    
    
    num = "Y"
    fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize = (18,5))
    fig.suptitle(cat+"   vs   "+"Y", fontsize=15)

    ### distribution
    if dtf[cat].dtype == "O":
        unq_cat = dtf[cat].unique().tolist()
        #unq_cat.sort()
        ax[0].title.set_text('density')
        for i in unq_cat:        
            sns.kdeplot(dtf[dtf[cat]==i][num], ax=ax[0], fill = cat)
            ax[0].grid(True)
    else:
        unq_cat = dtf[cat].unique().tolist()
        unq_cat.sort()
        for i in unq_cat:        
            sns.kdeplot(dtf[dtf[cat]==i][num], ax=ax[0], fill = cat)
            ax[0].grid(True)        
    
    ### stacked
    ax[1].title.set_text('bins')
    breaks = np.quantile(dtf[num], q=np.linspace(0,1,11))
    
    tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T    
    if dtf[cat].dtype == "O":
        tmp = tmp[dtf[cat].unique()]        
    tmp["tot"] = tmp.sum(axis=1)
    
    for col in tmp.drop("tot", axis=1).columns:
         tmp[col] = tmp[col] / tmp["tot"]
    tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
    ### boxplot   
    ax[2].title.set_text('outliers')
    sns.boxplot(x=cat, y=num, data=dtf, ax=ax[2])
    ax[2].grid(True)
    plt.show()
```
```ruby
widgets.interact(cat_vs_Y, cat = dtf[cat_cols])
```
Output: (Result Exemples)
<p align="center">
<img src=Images-plots/CY_Dependent_count.png />
</p>

This type of analysis are realy interesting to visually identify if there is some kind of pattern between a categorical feature and the target. For exemple, we can see that people with 0 dependents tend to concentrate on extremes of Y range. Sometimes we can have insights using this type of plots.
<br>
To complement this analysis, we can apply ANOVA tests in order to verify the relevance of these categorical features

``` ruby
# One-way ANOVA test
for cat in cat_cols:
    num = "Y"
    model = smf.ols(num+' ~ '+cat, data=dtf).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the", cat, "variable is", conclusion, "with Y (p-value: "+str(p)+")")
```
Output:
    Anova F: the Attrition_Flag variable is Non-Correlated with Y (p-value: 0.168)
    Anova F: the Gender variable is Non-Correlated with Y (p-value: 0.498)
    Anova F: the Dependent_count variable is Correlated with Y (p-value: 0.0)
    Anova F: the Education_Level variable is Non-Correlated with Y (p-value: 0.146)
    Anova F: the Marital_Status variable is Correlated with Y (p-value: 0.0)
    Anova F: the Income_Category variable is Correlated with Y (p-value: 0.014)
    Anova F: the Card_Category variable is Non-Correlated with Y (p-value: 0.484)
    Anova F: the Total_Relationship_Count variable is Non-Correlated with Y (p-value: 0.354)
    Anova F: the Months_Inactive_12_mon variable is Correlated with Y (p-value: 0.0)
    Anova F: the Contacts_Count_12_mon variable is Non-Correlated with Y (p-value: 0.278)

According to One-say ANOVA test, these are the variables with significant p-value (<= 0.05):

Dependent_count
Marital_Status 
Income_Category
Months_Inactive_12_mon 
Analysing "Marital_Status" visualization, categories doesn't seem to have different patterns in relation to "Y", so maybe we could drop it too.

**Numerical VS Y**
``` ruby
def num_vs_Y(x = dtf.columns[len(dtf.columns)-2]):    
    
    figsize = (18,4)
        
    y = "Y"
    ### bin plot
    dtf_noNan = dtf[dtf[x].notnull()]
    breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
    groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, 
               duplicates='drop')])[y].agg(['mean','median','size'])
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(x+"   vs   "+y, fontsize=15)
    groups[["mean", "median"]].plot(kind="line", ax=ax)
    groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True,
                        color="grey", alpha=0.3, grid=True)
    ax.set(ylabel=y)
    ax.right_ax.set_ylabel("Observazions in each bin")
    plt.show()
    ### scatter plot
    sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg')
    plt.show()
```
``` ruby
widgets.interact(num_vs_Y, x = dtf[num_cols])
```
Output: (Result Exemples)
<p align="center">
<img src=Images-plots/NY_customer_AgeXY.png />
</p>

**Insights**

1. Customer age seems to be directly correlated to Y

2. If we're trying to predic customers with only few "months on book", maybe "credit limit" between some intervals - (1762.0, 2787.0] and (3398.4, 4549.0] could be good indicators.

3. If we're trying to predic customers with only few "months on book", maybe "Total revolving bal" between (1037.4, 1276.0] could be good indicators.

4. If we're trying to predic customers with only few "months on book", maybe "Everage open to buy" between (1464.4, 3474.0] could be good indicators.

5. If we're trying to predic customers with only few "months on book", maybe "Total transaction amount" between (2411.0, 3192.4] could be good indicators.

6. If we're trying to predic customers with only few "months on book", maybe "Total transaction ct" between (41.0, 67.0] and (92.0, 139.0] could be good indicators.


Let's analyse Pearson Correlation test

``` ruby
for x in num_cols:
    y = "Y"
    dtf_noNan = dtf[dtf[x].notnull()]
    coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
    coeff, p = round(coeff, 3), round(p, 3)
    conclusion = "Significant" if p < 0.05 else "Non-Significant"
    if x != "Y":
        print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")", x)
```
Output:
    Pearson Correlation: 0.789 Significant (p-value: 0.0) Customer_Age
    Pearson Correlation: 0.008 Non-Significant (p-value: 0.45) Credit_Limit
    Pearson Correlation: 0.009 Non-Significant (p-value: 0.386) Total_Revolving_Bal
    Pearson Correlation: 0.007 Non-Significant (p-value: 0.498) Avg_Open_To_Buy
    Pearson Correlation: -0.049 Significant (p-value: 0.0) Total_Amt_Chng_Q4_Q1
    Pearson Correlation: -0.039 Significant (p-value: 0.0) Total_Trans_Amt
    Pearson Correlation: -0.05 Significant (p-value: 0.0) Total_Trans_Ct
    Pearson Correlation: -0.014 Non-Significant (p-value: 0.157) Total_Ct_Chng_Q4_Q1
    Pearson Correlation: -0.008 Non-Significant (p-value: 0.448) Avg_Utilization_Ratio

According to Person Correlation test, the significant variables are:

    Customer_Age
    Total_Amt_Chng_Q4_Q1
    Total_Trans_Amt
    Total_Trans_Ct
    
**Preprocessing data** 

Prepare raw data to make it suitable for a machine learning model

1. Each observation must be represented by a single row
2. Dataset must be partitioned into train and test data
3. Missing values must be replaced
4. Categorical data must be encoded (Bins)
5. Scale data

``` ruby
# store a list with all significant variables in one object
features = ["Customer_Age", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
            "Dependent_count", "Income_Category", "Months_Inactive_12_mon", "Y"]

dtf = dtf[features].copy()

```
That's how our dataset looks like after feature selection

Output:
<p align="center">
<img src=Images-plots/dataSet_afterFS.png />
</p>

**Feature engeneering**
<br>
First of all, lets define what we're trying to predict.
<br>
We want to identify clients who spend low periods on books and be able to directly adress them and try to convince them to stay longer. To that purpose we'll turn "Y" into a binary feature, being 1 good clients (>30 months on book) and 0 bad clientes (<= 30 months on book) 

``` ruby
for i in list(range(0,len(dtf))):
    if dtf.iloc[i, -1] > 30:
        dtf.iloc[i, -1] = 1
    else:
        dtf.iloc[i, -1] = 0
```

Let's see how many good and bad clients we have in our dataset

``` ruby
# good clientes
dtf["Y"].value_counts()
```
output: 
    1    7907
    0    2220
    Name: Y, dtype: int64
    
Notice that there are far more good clients than bad clientes and that's bad for our predictions. We can solve this problem by randomly selecting the same number of good and bad clientes.

``` ruby
# fixing the quantity problem
dtf = (dtf[dtf['Y'] == 0]).append(dtf[dtf["Y"] == 1].sample(n = dtf[dtf["Y"] == 0].count()[1]))
dtf = dtf.sample(n = len(dtf))
dtf.info()
```

``` ruby
# print the number of categories into categorical features
high_cat = []
n = 1
for i in features:    
    if (dtf[i].dtype == "O") & (dtf[i].nunique() >= n):
        print(i, "has", dtf[i].nunique(), "unique values") 
        cat = i
        high_cat.append(cat)
```
output:
    Income_Category has 6 unique values

``` ruby
# get dummies for Income_Category
dummy = pd.get_dummies(dtf["Income_Category"], drop_first = False)
dummy.drop("Unknown", axis = 1, inplace = True)
dtf = pd.concat([dtf.iloc[:, 0:(len(dtf.columns)-1)], dummy, dtf["Y"]], axis = 1)
dtf.drop("Income_Category", axis = 1, inplace = True)
```

Scale features using "RobustScaler"

``` ruby
## scale X
scalerX = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X = scalerX.fit_transform(dtf)
dtf_scaled = pd.DataFrame(X, columns=dtf.columns, index=dtf.index)
dtf_scaled["Y"] = dtf["Y"]

dtf_scaled.head()
```

That's how our dataset looks like after scaling

Output:
<p align="center">
<img src=Images-plots/dataSet_afterScalling.png />
</p>

**Feature Selection**

``` ruby
# correlation matrix
corr_matrix = dtf.corr(method="pearson")

fig, ax = plt.subplots(figsize=(18,7))
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=2, ax = ax)
plt.title("pearson correlation")
```

Output:
<p align="center">
<img src=Images-plots/Confusion_Matrix.png />
</p>

In this correlation matrix we can see that "Total_Trans_Amt" and "Total_Trans_Ct" may be explaining the same things. To avoid multicolinearity and to make our model as simple as possible, we will dive a little more into it.

Using ANOVA-f and RIDGE regularization (this second one is particularly useful to mitigate the problem of multicollinearity

``` ruby
X = dtf.drop("Y", axis=1).values
y = dtf["Y"].values
feature_names = dtf.drop("Y", axis=1).columns
## p-value
selector = feature_selection.SelectKBest(score_func=  
               feature_selection.f_regression, k=10).fit(X,y)
pvalue_selected_features = feature_names[selector.get_support()]

## regularization
selector = feature_selection.SelectFromModel(estimator= 
              linear_model.Ridge(alpha=1.0, fit_intercept=True), 
                                 max_features=10).fit(X,y)
regularization_selected_features = feature_names[selector.get_support()]
 
## plot
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["p_value"] = dtf_features["features"].apply(lambda x: "p_value" if x in pvalue_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in pvalue_selected_features else 0)
dtf_features["regularization"] = dtf_features["features"].apply(lambda x: "regularization" if x in regularization_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in regularization_selected_features else 0)
dtf_features["method"] = dtf_features[["p_value","regularization"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
dtf_features["method"] = dtf_features["method"].apply(lambda x: "both" if len(x.split()) == 2 else x)
sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)
```
Output:
<p align="center">
<img src=Images-plots/ANOVA&RIDGE.png />
</p>

The only four variables chosen by both methods (ANOVA-f and RIDGE) are: **Customer_Age | Total_Amt_Chng_Q4_Q1 | 40K - 60K | 60K - 80K | Less than $40K**
<br>
Now using ensemble methods to get feature importance (Gradient Boosting)

``` ruby
X = dtf.drop("Y", axis=1).values
y = dtf["Y"].values
feature_names = dtf.drop("Y", axis=1).columns.tolist()
## call model
model = ensemble.GradientBoostingRegressor()
## Importance
model.fit(X,y)
importances = model.feature_importances_
## Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE":importances, 
            "VARIABLE":feature_names}).sort_values("IMPORTANCE", ascending=False)
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")
    
## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize = (18,5))
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), 
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()
```
Output:
<p align="center">
<img src=Images-plots/Feature_importance.png />
</p>

We can clearly see that Customer_Age is the most important variable to the model. Considering the both methods of feature selection and the correlation matrix, we will run a first model using the following features:

    Customer_Age
    Total_Trans_Amt
    Total_Amt_Chng_Q4_Q1
    Total_Trans_Ct
    Months_Inactive_12_mon    
    
``` ruby
# select features
# names = ['Customer_Age', 'Months_Inactive_12_mon', 'Total_Trans_Ct', 'Total_Amt_Chng_Q4_Q1', "Y"]
names = ['Customer_Age', 'Total_Trans_Amt', 'Total_Amt_Chng_Q4_Q1', "Y"]
dtf_scaled = dtf_scaled[names]
```

Split data into training and test datasets

``` ruby
# split data
dtf_train, dtf_test = model_selection.train_test_split(dtf_scaled, test_size = 0.3)

## print info
print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())
```

``` ruby
X_train = dtf_train.drop("Y", axis = 1).values
y_train = dtf_train["Y"].values

X_test = dtf_test.drop("Y", axis = 1).values
y_test = dtf_test["Y"].values
```
## Model design
Run a simple linear regression to use as base for a more complex one, as Gradient Boosting

``` ruby
## K fold validation
## call models to be compared
model_XGboost_class = xgboost.XGBClassifier()
model_LR = linear_model.LogisticRegression(solver="liblinear", random_state=0)
model_SVM = svm.SVC(kernel="linear", probability=True, random_state=0)
# "model_XGboost_class", "model_LR", 
lst_models = ["model_SVM", "model_XGboost_class", "model_LR"]

def model_func(model):
    if model == "model_XGboost_class":
        model_fit = model_XGboost_class    
    elif model == "model_LR":
        model_fit = model_LR 
    elif model == "model_SVM":
        model_fit = model_SVM
        
    cv = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize = (15,5))
    fig.suptitle((model), fontsize=20)
    
    # ROC curve
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model_fit.fit(X_train[train], y_train[train])
        viz = RocCurveDisplay.from_estimator(
            model_fit,
            X_train[test],
            y_train[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax = ax[0]
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax[0].plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax[0].plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax[0].fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev."
    )

    ax[0].set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic"
    )
    ax[0].legend(loc="lower right")
    
    # Confusion matrix
    y_pred = model_fit.predict(X_train[test])

    cf_matrix = metrics.confusion_matrix(y_train[test], y_pred)


    ax[1] = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap='Blues', fmt='.2%')

    ax[1].set(title = ('Confusion Matrix ('+ str(round(100*(cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix), 2)) + '% accuracy)'));
    ax[1].set_xlabel('Predicted Values')
    ax[1].set_ylabel('Actual Values');

    ## Ticket labels - List must be in alphabetical order
    ax[1].xaxis.set_ticklabels(['False','True'])
    ax[1].yaxis.set_ticklabels(['False','True'])
    
    # Plot chart
    plt.show()      
```
Output:
<p align="center">
<img src=Images-plots/SVM_result.png />
</p>
<p align="center">
<img src=Images-plots/XGBoost_model.png />
</p>
<p align="center">
<img src=Images-plots/LR_results.png />
</p>

### Conclusion
This article validated the importance of a good feature selection and how it can simplify a model structure and improve its final results.

Speaking about results, it can be seen that all three models performed, but, in this case, Linear Regression and SVM has presented a better mean ROC and accuracy when classifing clients types. As, in theory, Linear Regression is a simpler model then SVM, it should be priorized in this type of problems. 

Thanks for reading. If you have any considerations about this, please, send me a message. 

Best regards!


