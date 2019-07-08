Title: Pipelines
Date: 2019-07-03 16:40
Tags: python
Slug: blog-2

Pipelines are great tools for duplicating efforts you’ve already made against a dataset, making them available to new information. It wraps up your data cleaning, your model, and any other intermediary steps all in one ordered operation.

Let's start with a very simple data frame to work on.


```python
import pandas as pd
import numpy as np
```


```python
df = pd.DataFrame({
    'age': [25,22,26,np.nan,30,35,40,42,43,44],
    'income': [40,37,42,60,58,70,62,85,120,95],
    'owns_car': [0,0,1,0,0,1,1,0,1,1]
})
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
      <th>owns_car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.0</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>37</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>42</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30.0</td>
      <td>58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>40.0</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42.0</td>
      <td>85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>43.0</td>
      <td>120</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>44.0</td>
      <td>95</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We will be predicting car ownership so let's split the features from the target now.


```python
features = ['age','income']
X = df[features]
y = df['owns_car']
```

To begin lest go through some of the usual cleaning suspects to get this ready for a logistic regression. Specifically lets bring in SimpleImputer and StandardScaler.


```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
```

To start let's deal with the NaN value in Age. SimpleImputer will handle imputing the mean age for that value. We are then stuffing that all back into a DataFrame so we can see what we've done.


```python
imp = SimpleImputer(missing_values=np.nan,strategy='mean', copy=True)
imputed = pd.DataFrame(imp.fit_transform(X))
imputed.columns = X.columns
X = imputed
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.000000</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.000000</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.000000</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34.111111</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30.000000</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.000000</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>40.000000</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42.000000</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>43.000000</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>44.000000</td>
      <td>95.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's bring in StandardScaler to scale our features. Again we stuff it back into a DataFrame for ease of viewing.


```python
scal = StandardScaler()
scaled = pd.DataFrame(scal.fit_transform(X))
scaled.columns = X.columns
X = scaled
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.189305e+00</td>
      <td>-1.068765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.580906e+00</td>
      <td>-1.187959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.058772e+00</td>
      <td>-0.989303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.274965e-16</td>
      <td>-0.274144</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-5.366378e-01</td>
      <td>-0.353606</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.160298e-01</td>
      <td>0.123166</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.686974e-01</td>
      <td>-0.194682</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.029764e+00</td>
      <td>0.719132</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.160298e+00</td>
      <td>2.109719</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.290831e+00</td>
      <td>1.116443</td>
    </tr>
  </tbody>
</table>
</div>



We now have happy data prepared for the vast majority of algorithms you may want to throw at it. We are going to throw a default logistic regression on it, using the features in X to predict car ownership.


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression(random_state=42, solver='lbfgs')
lr.fit(X,y)
lr.score(X,y)
```




    0.8



Now we have a working model, maybe not the most accurate, but it does run. If we wanted to apply this model to a new piece of data we would need to run through each of those steps, making transformations against the new data. That seems like a hassle. That is a hassle. Instead below I am going to bring in pipeline to solve this. First I am going to re-instantiate our original feature data as bf.


```python
bf = pd.DataFrame({
    'age': [25,22,26,np.nan,30,35,40,42,43,44],
    'income': [40,37,42,60,58,70,62,85,120,95]
})
```

Now that we have some "New" raw data to have the model predict against we will create the Pipeline. It consists of a list of tuples it will run in order. The first half of the tuple is always the name of the step, the second is the function it will execute.


```python
from sklearn.pipeline import Pipeline
```


```python
pipe = Pipeline([
        ('simple_imputer', SimpleImputer(missing_values=np.nan,strategy='mean')),
        ('scale', StandardScaler())
])
```

With pipe now instantiated we can use it to transform bf to align with what our model needs. Make sure the fit happens against original data and not against the new data, as this will alter how the scaling and average used in our cleaning.


```python
pipe.fit(df[features])
piped_bf = pipe.transform(bf)
piped_bf = pd.DataFrame(piped_bf)
piped_bf.columns = bf.columns
piped_bf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.189305e+00</td>
      <td>-1.068765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.580906e+00</td>
      <td>-1.187959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.058772e+00</td>
      <td>-0.989303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.274965e-16</td>
      <td>-0.274144</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-5.366378e-01</td>
      <td>-0.353606</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.160298e-01</td>
      <td>0.123166</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.686974e-01</td>
      <td>-0.194682</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.029764e+00</td>
      <td>0.719132</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.160298e+00</td>
      <td>2.109719</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.290831e+00</td>
      <td>1.116443</td>
    </tr>
  </tbody>
</table>
</div>



Now we can use our original logistic regression to predict against the new data.


```python
lr.predict(piped_bf)
```




    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])



Now what has been done above works, but it's not aligned with how we would actually get this going in production. Train-test-split was not really executed, and a sample size of 10 is easy to demonstrate on but very unrealistic. As well Pipeline gives us the ability to wrap the model within the pipeline and we should look at that. 

Below I am going to lay out how I actually implement Pipeline. To do this I will be bringing in DataFrameMapper from sklearn_pandas, I will not get verbose on how to use mapper, but I would highly suggest you take a look at its documentation as it is a great tool. I am also going to sample our data out to get up to a number that seems a bit more realistic for running this sort of prediction.


```python
from sklearn.utils import resample
```


```python
cf = resample(df,n_samples=500_000,random_state=42)
cf = cf.reset_index()
cf.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>age</th>
      <th>income</th>
      <th>owns_car</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>499995</th>
      <td>3</td>
      <td>NaN</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499996</th>
      <td>0</td>
      <td>25.0</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499997</th>
      <td>8</td>
      <td>43.0</td>
      <td>120</td>
      <td>1</td>
    </tr>
    <tr>
      <th>499998</th>
      <td>7</td>
      <td>42.0</td>
      <td>85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499999</th>
      <td>7</td>
      <td>42.0</td>
      <td>85</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have it sampled out to 500,000 we will train test split, features is still defined from much earlier in the code, and standard train test split parameters will be used.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cf[features], cf['owns_car'], random_state=42)
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>359342</th>
      <td>22.0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>236051</th>
      <td>30.0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>452617</th>
      <td>43.0</td>
      <td>120</td>
    </tr>
    <tr>
      <th>34245</th>
      <td>25.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>373935</th>
      <td>43.0</td>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn_pandas import DataFrameMapper
```


```python
mapper = DataFrameMapper([
    (['age'], [SimpleImputer(missing_values=np.nan,strategy='mean'),StandardScaler()]),
    (['income'], StandardScaler())
])
```

Now that we have mapper set up we will make a new pipeline, pipe_2, and stuff it with the mapper and the logistic regression. Pipeline always executes in order, so the mapper will clean our data up and the regression will then be applied. If we needed extra steps, say some PCA, we can just slot it in after the cleaning, because again it always executes in order. 


```python
pipe_2 = Pipeline([
    ('map', mapper),
    ('log', LogisticRegression(random_state=42, solver='lbfgs'))
])
```

Now it's time to execute pipe_2. First we fit it against the training set, this will establish the cleaning parameters and create the logistic model for us to predict off of. Next we have pipe_2 predict against X_test and then compare these predictions to y_test.


```python
pipe_2.fit(X_train, y_train)
y_pred = pipe_2.predict(X_test)
comp = pd.DataFrame({
    'y_pred': y_pred,
     'y_test': y_test
})
comp.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_pred</th>
      <th>y_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>104241</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>199676</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140199</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>132814</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>408697</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163280</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>215758</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>442316</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6940</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>382310</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>472236</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>309086</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>230672</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>209236</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102953</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Not perfect, but it's clearly working. Pipeline makes your flow much more modular. If you need more steps you slot them in the pipeline where you want them to happen. If you need a new model you swap it out. Better yet Pipeline can take pipelines within them. Have a weird replace-NA to Lambda to regularization function to fix a feature before it hits mapper? No problem, wrap it up and throw it in the pipeline before mapper. 

As stated at the top Pipeline helps wrap your process all together. It's the box you can throw everything into. Hopefully it will let you be a bit more poetic. Throughout this I have shown you Pipeline, which requires a tuple of a title and a function. There is another option in make_pipeline, which doesn't require a title, just the function, which it arbitrarily names for you. Most days you will end up using make_pipeline, as it is faster. But when you need more specific documenting, Pipeline will be the ticket.