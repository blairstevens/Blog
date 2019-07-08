Title: Lambda Functions
Date: 2019-06-18 1:45
Tags: python
Slug: blog-1

Lambda functions are an easy way to quickly define functions within Python. They are a departure from the regular def format for creating functions and can be confusing at first. Let’s begin with defining a simple square root function using the normal def syntax, and then converting it into a lambda function.


```python
import numpy as np

def root(x):
    return np.sqrt(x)

```

In this instance any time we call root it will return the square root of x.


```python
root(16)
```




    4.0



Now we are going to replicate root as lamroot, where lamroot is defined using lambda.


```python
lamroot = lambda x: np.sqrt(x)
```

Once again calling lamroot will return the square root of x.


```python
lamroot(16)
```

Lets piece apart what has gone on here.

Lambda lets us define an outcome against a parameter with some sort of function, in this case x is the parameter, and NumPy’s square root function is being called against x to create the outcome. The split between the parameter and the function is the colon, everything before are parameters, everything after is the manipulations the function is making.

We can define a lambda function with more than one parameter.



```python
lamroot2 = lambda x,y: np.sqrt(x+y)
```

Here we have given the function two parameters, x and y, and it will return the square root of the sum of the two parameters.


```python
lamroot2(10,6)
```




    4.0



From the structure we have laid out so far you are able to replace many functions throughout your code with lambda functions. There is a drawback to lambda functions in they do sacrifice some clarity in exchange for brevity. It is up to you to determine which is more useful in your situation.

Lambda functions really come into there own when applied to collections of data, such as lists or pandas DataFrames. Instead of writing a loop to manipulate these sorts of collections we can instead write a map or apply statement against the collection to have every piece calculated at once.

To demonstrate this I’m going to import pandas and spin up a quick DataFrame with the integers from 0 to 10.


```python
import pandas as pd

df = pd.DataFrame(np.arange(0,11))

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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



Now we rewrite the previous lamroot function within an apply function on the DataFrame.


```python
df = df.apply(lambda x: np.sqrt(x))
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.189207</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.316074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.495349</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.565085</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.626577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.681793</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.732051</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.778279</td>
    </tr>
  </tbody>
</table>
</div>



Here we have applied the function against the entire DataFrame, viewing it now will show our integers have been turned into their square roots. This is very useful to write quick single line functions to cleanup or clarify a specific aspect of a dataset.