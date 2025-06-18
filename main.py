# https://www.youtube.com/watch?v=XVsYazU_M2c

import NeuralNN
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset
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

#print(f"Data to use for a INT&OUT {df_data_pairs}")
sns.catplot(data=df_x).set_xticklabels(rotation=30, labels=feature_name).set_ylabels('[cm]')            # Show a basic data 

dataScaler = StandardScaler()                                                                           # Scaled data
x_scaled = dataScaler.fit_transform(x)

encoding = OneHotEncoder(sparse_output=False)                                                           # One hot encoding
y_onehot = encoding.fit_transform(y.reshape(-1, 1))
#y_onehot

df_x_scaled = DataFrame(x_scaled)
sns.catplot(data=df_x_scaled).set_xticklabels(rotation=30, labels=feature_name).set_ylabels('[cm]')      
#plt.show()

# REGION TRAINING

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, test_size=0.2, random_state=42) # SPLIT DATA

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)                                             # CHANGE TO TENSOR 
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)                                           # DATA'S TRAIN
train_load = DataLoader(train_dataset, batch_size=16, shuffle=True)                                     # LOADER DATA TARIN


model = NeuralNN.NeuralNN()

criterion = nn.CrossEntropyLoss()                                      # function of error
optymalizer = optim.Adam(model.parameters(), lr=0.001)                 # optimalizer


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_load:
        optymalizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optymalizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Interaction number: [{epoch+1}/{num_epochs}] Net error: {loss.item():.4f}')

model.eval()
while torch.no_grad():
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, dim=1)
    _, y_test_labels = torch.max(y_test_tensor, dim=1)
    accuracy = (predicted == y_test_labels).sum().item() / y_test_labels.size(0)
    print(f'Accuracy of model on base tets data : correct: {accuracy * 100:.2f}%')

# tourch.save(model, myModel.pth)

# training_model = torch.load('myMOdel.pth')
# outputs = trained_model(x_test_tensor)
# outputs