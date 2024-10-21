from tabulate import tabulate
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
df = pandas.read_csv('titanic.csv')
a =['Cabin','Name','PassengerId','Ticket','Embarked']
df.drop(a,axis=1,inplace=True)
def change(sex):
    if sex == 'male':
        sex = 0
        return int(sex)
    if sex == 'female':
        sex = 1
        return int(sex)
df['Sex']= df['Sex'].apply(change)


age1 = df[df['Pclass'] == 1]['Age'].median()
age2 =  df[df['Pclass'] == 2]['Age'].median()
age3 =  df[df['Pclass'] == 3]['Age'].median()

def change2(age):
    if pandas.isnull(age['Age']):
        if age['Pclass']==1:
            return age1
        if age['Pclass'] == 2:
            return age2
        if age['Pclass'] == 3:
            return age3
    return age['Age']








df['Age']= df.apply(change2,axis=1)
# print(tabulate(df,headers = 'keys',tablefmt='pretty'))













x = df.drop('Survived',axis=1)
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classfier = KNeighborsClassifier(n_neighbors=5)
classfier.fit(x_train,y_train)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = classfier.predict(x_test)
# print(accuracy_score(y_test,y_pred)*100)


def new_passenger(pclass, sex, age, sibsp, parch, fare):
    data_passenger = [[pclass, sex, age, sibsp, parch, fare]]
    data_passenger = sc.transform(data_passenger)
    sur = model.predict_proba(data_passenger)[0][1]
    return sur


print(new_passenger(2,1,12,0,1,3)*100)
