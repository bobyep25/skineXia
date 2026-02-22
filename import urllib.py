import urllib
page = urllib.urlopen('index.html').read()
print(page)