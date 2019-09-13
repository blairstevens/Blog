Title: parse_it
Date: 2019-08-07 12:52
Tags: python
Slug: blog-5

Configuration files are either something you hardly ever use, or they're something you wrestle with. If you're in the former camp you should start using them to make your models distributable and secure, and consider parse_it to do it. If you're in the latter camp you're here to see what parse_it can do for you.<br/><br/>
Let's start with a little example, bringing in os to build the environment and parse_it to read it.


```python
import os
os.environ["IMPORTANT_WORD"] = "ThIsIsAwOrD"
os.environ["SEVERAL_THINGS"] = "['the', 'last', 'one', 'will', 'be', 'a', 'number', '42']"
os.environ["ALSO_DICTIONARIES"] = "{'oxford': 'webster'}"
```

Now we have some random configuration stuff spun up. Let's peel them out using parse_it.


```python
from parse_it import ParseIt

parser = ParseIt()

my_word = parser.read_configuration_variable("IMPORTANT_WORD")
my_word
```




    'ThIsIsAwOrD'




```python
my_list = parser.read_configuration_variable("SEVERAL_THINGS")
my_list
```




    ['the', 'last', 'one', 'will', 'be', 'a', 'number', 42]




```python
my_dict = parser.read_configuration_variable("ALSO_DICTIONARIES")
my_dict
```




    {'oxford': 'webster'}



I realize at this point that those of you who do consistently use config files are note impressed at all with the local and spin up I've just done. And that is fair, it isn't very impressive. The fact of the matter is that showing well constructed config architectures would make for a lengthy and tutorial like blog. Even if I were to write that piece it would only scratch the surface on what the options are for architectures and formats.<br/><br/>
This is where parse_it really shines. As long as the user you're distributing to can place a file within the designated folder you are set. Parse_it reads 13 different file types and apply their values within your script. This minimizes the formatting and filetype issues we are all used to. This isn't a new engine powering your configs, instead it's a new oil to make them run that much better, and I suggest you give it a try.


```python

```