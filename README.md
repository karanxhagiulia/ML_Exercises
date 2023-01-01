# ML_Exercises
## Loan prediction
- Loan Prediction Exercise with Random Forest and Pipeline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karanxhagiulia/ML_Exercises/blob/main/LOAN_Exercise_Random_Forest.ipynb)

Loan prediction
===============

  

[To open in colab click here](https://colab.research.google.com/github/karanxhagiulia/ML_Exercises/blob/main/LOAN_Exercise_Random_Forest.ipynb)

  

Cleaning
--------

  

EN: For self\_employed, we have 500 "no", 82 "yes". I can either use the mode to fill the null values, or check by myself which values are closer. In this case, it's more likely that the values are a "no".

IT: Per self\_employed, dato che 500 non sono self employed e 82 si, controllo le statistiche per i valori nan e decido se rimpiazzarli con "No", dato che è più probabile che non siano self employed visti i numeri

  

```plain
df['Self_Employed'] = df['Self_Employed'].fillna('No')
```

  

  

EN: My null values in the Gender feature are only 13 and very different from the two genders: I decide to drop them. Otherwise, I could have used the mode in this case too.

  

IT: Decido di eliminare i record con valori nulli in Gender: hanno dei valori troppo alti e diversi dagli altri due, e sono solo 13.

  

```plain
df= df.dropna(subset = ['Gender']) 
```

Label encoding
--------------

  

Since 3+ is a string, I have to change it to an int

  

```plain
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype('int')

from sklearn.preprocessing import LabelEncoder #I'm using the Label Encoder for my target
enc = LabelEncoder()

df['Loan_Status'] = enc.fit_transform(df['Loan_Status'])
enc_name_mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))
print(enc_name_mapping) #this is the dictionary with the values of my target
```

Categorical features
--------------------

  

```plain
categorical_features = df[['Gender', 'Married', 'Education','Self_Employed',
       'Property_Area']] #cat featu without target

for col in categorical_features:
    print(df[col].unique())
```

`['Male' 'Female']`

`['No' 'Yes']`

`['Graduate' 'Not Graduate']`

`['No' 'Yes']`

`['Urban' 'Rural' 'Semiurban']`

  

  

I'll use map to change the categorical into numerical values:

  

```plain
df['Gender']= df['Gender'].map({'Male':0, 'Female':1}) 
```

  

I'll save them as dictionaries so I can have a legend:

  

```plain
Gender = {'Male':0, 'Female':1}
```

EDA
---

![image](https://user-images.githubusercontent.com/96819403/210185755-87800b88-d386-4ac5-ba65-3cbc5fd2159e.png)

![image](https://user-images.githubusercontent.com/96819403/210185778-95f22f21-06a6-4b06-bc8b-196d62f635d2.png)

![image](https://user-images.githubusercontent.com/96819403/210185799-1d4d0d7f-9db2-43cc-b503-f78e207e0235.png)

Train Test 
---
![image](https://user-images.githubusercontent.com/96819403/210185817-f07a6c0a-ddf2-4104-81fb-c8a3efb95674.png)
![image](https://user-images.githubusercontent.com/96819403/210185830-e89f15c8-0038-4965-99aa-5347f301160d.png)

Model Evaluation 
---
![image](https://user-images.githubusercontent.com/96819403/210185842-0ad6bb0e-0c17-4660-b12c-3bb847692872.png)

