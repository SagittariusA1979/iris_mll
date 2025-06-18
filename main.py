# https://www.youtube.com/watch?v=XVsYazU_M2c

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_iris
from pandas import DataFrame



iris = load_iris()                    # We get a data abouth iris
x = iris.data                         # Feature of .... - input
y = iris.target                       # Type based on features .... - output


feature_name = iris.feature_names
target_name = iris.target_names

df_x = DataFrame(x)
df_x_label = df_x.set_axis([feature_name[0], feature_name[1], feature_name[2], feature_name[3]], axis=1)

df_y = DataFrame(target_name[y])
df_data_pairs = df_x_label
df_data_pairs['iris_type'] = df_y

print(f"Data to use for a INT&OUT {df_data_pairs}")
sns.catplot(data=df_x).set_xticklabels(rotation=30, labels=feature_name).set_ylabels('[cm]')            # Show a basic data 

dataScaler = StandardScaler()                                                                           # Scaled data
x_scaled = dataScaler.fit_transform(x)

encoding = OneHotEncoder(sparse_output=False)                                                           # One hot encoding
y_onehot = encoding.fit_transform(y.reshape(-1, 1))
y_onehot

df_x_scaled = DataFrame(x_scaled)
sns.catplot(data=df_x_scaled).set_xticklabels(rotation=30, labels=feature_name).set_ylabels('[cm]')      
#plt.show()

# REGION TRA

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, test_size=0.2, random_state=42)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
