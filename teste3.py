# %% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p = np.linspace(0,20, 100)
plt.plot(p, np.sin(15*p))
plt.plot()
plt.show()
# %%
import seaborn as sns

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
sns.lineplot(x = x, y= y**2)
plt.show()
# plt.plot(x, np.sin(x))
# plt.show()
# %%
print("teste")

print("teste2")

# %%
