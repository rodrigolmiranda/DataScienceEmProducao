import numpy as np
import requests
import seaborn as sns
print("aaaa")
print("asd")
response = requests.get("https://randomuser.me/api?results=10")
data = response.json()
for user in data['results']:
    print(user['name'] ['first'])

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
sns.lineplot(x = x, y= y)