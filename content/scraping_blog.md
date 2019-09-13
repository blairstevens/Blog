Title: Web Scraping
Date: 2019-07-30 12:55
Tags: python
Slug: blog-3

Somewhere in your data science journey you will need to scrape a webpage. This post will try to get you up and running as fast as possible. 

To start we need to import requests to look at the webpage, and BeautifulSoup to work with that data.


```python
import requests
from bs4 import BeautifulSoup
```

Now we are going to identify a page to scrape. Our target is going to be band members from their Discogs page. In this case it's going to be Larkin Poe because they are awesome and you should be listening to them. We will stuff the url into a variable and then use request.get to retrieve the data.


```python
url = 'https://www.discogs.com/artist/2487986-Larkin-Poe'

resp = requests.get(url)

page = resp.text
```

Now we have stuffed the HTML code into page. You may want to take a look through the full page code to familiarize yourself with some html, however if I were to do that this would become a very long blog post. Instead I have printed the first thousand characters below to give you a taste.


```python
print(page[0:1000])
```

    <!DOCTYPE html>
    <html
        class="is_not_mobile needs_reduced_ui "
        lang="en"
        xmlns:og="http://opengraphprotocol.org/schema/"
        xmlns:fb="http://www.facebook.com/2008/fbml"
    >
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta http-equiv="content-language" content="en">
            <meta http-equiv="pragma" content="no-cache" />
            <meta http-equiv="expires" content="-1" />
    
                    <meta id="viewport" name="viewport" content="initial-scale=1.0,width=device-width" />
    
            <script>
                if (window.innerWidth <= 600) document.getElementById('viewport').setAttribute('content', 'initial-scale=1.0');
                                                </script>
            
            <script>
                window.__start = new Date().getTime();
            </script>
    
                        <meta name="description" content="Explore releases and tracks from Larkin Poe at Discogs. Shop for Vinyl, CDs and more from Larkin Po


From here we are going to pass page into BeautfulSoup which is a package that helps us pull data from the html we brought in with request. The second argument we pass here is the parser we want BeautifulSoup to use. In some circumstances you may need to change your parser, the BeautifulSoup documentation does a good job of laying out those cases. I should emphasize that if you do swap out your parser you need be consistent across environments and implementations to ensure consistent results. From here we can pull different aspects of the code, in this case we will print the page title html code.


```python
soup = BeautifulSoup(page, 'html.parser')
print(soup.title)
```

    <title>Larkin Poe | Discography &amp; Songs | Discogs</title>


From here we can use BeautifulSoup's find_all method to pull out each line with class tag readmore. The easiest way to find out what you need to search for is by using your browser to inspect the area of the page you are attempting to pull data from. This is usually as easy as right click and inspect on the element. The find_all method can locate on several different features of the lines within html code. I would suggest reading through the find_all documentation to get a better idea of what it can fixate on.


```python
found = soup.find_all(class_='readmore')
print(found)
```

    [<div class="readmore" id="profile">
                Rebecca &amp; Megan Lovell
                        </div>, <div class="readmore">
    <a href="/artist/4443794-Megan-Lovell">Megan Lovell</a>, 
                        <a href="/artist/3126405-Rebecca-Lovell">Rebecca Lovell</a> </div>]


We now have drawn a block of code that definitely contains the band member names, but also is still bogged down by html tags and formatting. Here we will import regular expressions, and use it to bring out the text sections with their names. Here it is fixating on strings that have capital letters. Regular expressions are a deep and convoluted subject, with many resources online to help you. I end up leaning on these resources every time I write one, and suggest you do as well.


```python
import re
regular = re.findall(r'\b[A-Z].*', (str(found)))
print(regular)
```

    ['Rebecca &amp; Megan Lovell', 'Megan-Lovell">Megan Lovell</a>, ', 'Rebecca-Lovell">Rebecca Lovell</a> </div>]']


We are down to an easy looking list, the last two elements have our names, the first is a combination of both and not useful to us. Here I am going to define a function that will loop through the list and pull out the names. We will select only the last two elements by looking for '>' in the strings, and then splitting on that character. It also indexes to the first element of the split which gives us the name with a "-" in the middle. This is appended to a new list, and then that list is run through a loop that pulls the dash and replaces it with a space, as well as cleaning up an errant quote. The results of this are compiled into a list and then returned.


```python
def cleaner(in_list):
    a = []
    b = []
    for i in regular:
        if '>' in i:
            a.append((i.split('>',1))[0])
        else:
            pass
    for i in a:
        c = i.replace('"',"")
        b.append((c.replace("-"," ")))
    return b
```


```python
done = cleaner(regular)
done
```




    ['Megan Lovell', 'Rebecca Lovell']



We now have a list of the band members. Take this code and try it out on your favourite band. Play around with it and see if you can scrape other fields about the band. To extend this further take a look at the scrapy package and see if you can get a crawler going to scrape the top bands on discogs.