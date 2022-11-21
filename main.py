import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

model = SVC()
model_linear_kernal = SVC(kernel='linear')

labelEncoder = LabelEncoder()

df = pd.read_csv('Loan_Data.csv', sep=",")

df_sorted = df.sort_values(by="Loan_Status")
df_sorted = df_sorted.drop(['Loan_ID'], axis='columns')

print("Loan Status Before" + str(df_sorted['Loan_Status'].unique()))
print("Gender Before" + str(df_sorted['Gender'].unique()))
print("Married Before" + str(df_sorted['Married'].unique()))
print("Dependents Before" + str(df_sorted['Dependents'].unique()))
print("Education Before" + str(df_sorted['Education'].unique()))
print("Self Employed Before" + str(df_sorted['Self_Employed'].unique()))
print("Property Area Before" + str(df_sorted['Property_Area'].unique()))

df_sorted['Loan_Status'] = labelEncoder.fit_transform(df_sorted['Loan_Status'])
df_sorted['Gender'] = labelEncoder.fit_transform(df_sorted['Gender'])
df_sorted['Married'] = labelEncoder.fit_transform(df_sorted['Married'])
df_sorted['Dependents'] = labelEncoder.fit_transform(df_sorted['Dependents'])
df_sorted['Education'] = labelEncoder.fit_transform(df_sorted['Education'])
df_sorted['Self_Employed'] = labelEncoder.fit_transform(df_sorted['Self_Employed'])
df_sorted['Property_Area'] = labelEncoder.fit_transform(df_sorted['Property_Area'])

print("Loan Status After" + str(df_sorted['Loan_Status'].unique()))
print("Gender After" + str(df_sorted['Gender'].unique()))
print("Married Before" + str(df_sorted['Married'].unique()))
print("Dependents Before" + str(df_sorted['Dependents'].unique()))
print("Education Before" + str(df_sorted['Education'].unique()))
print("Self Employed Before" + str(df_sorted['Self_Employed'].unique()))
print("Property Area Before" + str(df_sorted['Property_Area'].unique()))

df_sorted_Y = df_sorted[192: ]
df_sorted_N = df_sorted[:192]

print(tabulate(df_sorted, headers = 'keys', tablefmt = 'psql'))

# plt.scatter(df_sorted_N['ApplicantIncome'], df_sorted_N['Credit_History'],color="green",marker='+')
# plt.scatter(df_sorted_Y['ApplicantIncome'], df_sorted_Y['Credit_History'],color="blue",marker='.')

# plt.show()

X = df_sorted.drop(['Loan_Status'], axis='columns')
y = df_sorted.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model_linear_kernal.fit(X_train, y_train) #Doesn't take in NaN values, is there a way I can just add in random values for now?
print(model_linear_kernal.score(X_test, y_test))
#model_linear_kernal.predict([[]]]) add values to predict here