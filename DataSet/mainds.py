# https://www.youtube.com/watch?v=PM8udzOFPEM

import tabulate

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


def show(dataSet, numCol):              # Show data and preparing 
    #for i, col in enumerate(dataSet):
    #    print(f'{i}: {col}')

    #pd.set_option("display.max_columns", None)   # Show all columns
    pd.set_option("display.width", 200)          # Widen the output to fit terminal
    #pd.set_option("display.max_colwidth", None)  # Show full column content

    print(tabulate.tabulate(dataSet.head(10), headers='keys', tablefmt='psql'))
def parseArea(val):                     # removed and change a specicfity data 
              if pd.isna(val) or val == '':
                      return np.nam
              
              val = str(val).replace(',', '.')
              textTolowerCase = str(val).lower()
              textOnlyDigit = re.sub(r'[0-9\.]', '', valueToChange ) # Fake


              try:
                      areaVal = float(val)

                      if areaVal < 0:
                            return np.nan
                      return areaVal
              except ValueError:
                      return np.nan             
def dropNaN (prease_build_data):        # We remover all [nan] fro dataSet
        df['bulidData'] = df['built_data'].apply(prease_build_data)
        



df = pd.read_csv('ElVehicle.csv', sep=',', decimal='.')

#print(df.info())
#print(df.head())


#df = df[df['valueToRemove'] >= 100]         # data which I can remowe
#df['area'] = df['area'].apply(parse_area)   # apply
largest_ElectricRange = df.loc[df['Electric Range'].idxmax()]     # max value
smallest_ElectricRange = df.loc[df['Electric Range'].idxmin()]    # min value


# normalization & standardization [ engineering of feature ]

scalerA = MinMaxScaler()    # standarization
charS = scalerA.fit_transform(df[['Electric Range', 'DOL Vehicle ID']])
#df[['Electric Range', 'DOL Vehicle ID']] = charS


scalerB =  StandardScaler() # Normalization
charN = scalerB.fit_transform(df[['Electric Range', 'DOL Vehicle ID']])
#df[['Electric Range', 'DOL Vehicle ID']] = charN

# CHARs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(df['Electric Range'], df['DOL Vehicle ID'], color='blue', alpha=0.7)
axes[0].set_title('Ogrinal data')
axes[0].set_xlabel('Electric Range')
axes[0].set_ylabel('DOL Vehicle ID')

axes[1].scatter( charS[:, 0], charS[:, 1], color='green', alpha=0.7)
axes[1].set_title('Standarization')
axes[1].set_xlabel('Electric Range [STR]')
axes[1].set_ylabel('DOL Vehicle ID [STR]')

axes[2].scatter( charN[:, 0], charN[:, 1], color='orange', alpha=0.7)
axes[2].set_title('Standarization')
axes[2].set_xlabel('Electric Range [NOR]')
axes[2].set_ylabel('DOL Vehicle ID [NOR]')



plt.show()



#print(df.sample(10))


#df =  df.sort_values(by='VIN (1-10)')
#df.drop_duplicates(subset=['Electric'], keep='last', inplace=True)







