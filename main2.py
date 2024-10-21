import pandas
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
x = pandas.read_csv('train.csv')
y = pandas.read_csv('test.csv')
x = x.drop(['id','langs','city','career_start','career_end','occupation_name','people_main','life_main','graduation'],axis=1)
y = y.drop(['id','langs','city','career_start','career_end','occupation_name','people_main','life_main','graduation'],axis=1)
def change(octype):
        if octype == 'university':
            octype = 0
            return int(octype)
        if octype == 'work':
            octype = 1
            return int(octype)
        else:
            octype = 0
            return int(octype)

def change2(ed_st):
    if ed_st == 'Alumnus (Specialist)':
        ed_st = 0
        return int(ed_st)
    if ed_st == 'Student (Specialist)':
        ed_st = 1
        return int(ed_st)
    if ed_st == "Student (Bachelor's)":
        ed_st = 2
        return int(ed_st)
    if ed_st =="Alumnus (Bachelor's)":
        ed_st = 3
        return int(ed_st)
    if ed_st == "Alumnus (Master's)":
        ed_st = 4
        return int(ed_st)
    if ed_st == "PhD":
        ed_st = 5
        return int(ed_st)
    if ed_st == "Student (Master's)":
        ed_st = 6
        return int(ed_st)
    if ed_st == 'Undergraduate applicant':
        ed_st = 7
        return int(ed_st)
    if ed_st == 'Candidate of Sciences':
        ed_st = 8
        return int(ed_st)

def change3(ed_form):
    if ed_form == 'Full-time':
        ed_form = 0

    elif ed_form == 'Distance Learning':
        ed_form = 1
    elif ed_form == 'Part-time':
        ed_form = 2
    elif ed_form =='External':
        ed_form = 3
    else:
        ed_form = 0
    return ed_form

def change_time(time):
    time = time.split(' ')
    time = time[0]
    time = time.split('-')
    month = int(time[1])
    year = int(time[0])
    if month == 12 or month == 11 and year == 2020:
        return 1
    else:
        return 0


def change_age(age):
    try:
        age = age.split('.')

    except:
        age = 1990


    try:
        age = age[2]
    except:
        age = 1990
    return age





x['bdate'] = x['bdate'].apply(change_age)
x['last_seen'] = x['last_seen'].apply(change_time)
x['education_form'] = x['education_form'].apply(change3)
x['education_status'] = x['education_status'].apply(change2)
x['occupation_type']= x['occupation_type'].apply(change)
# print(tabulate(x.head(10),headers = 'keys',tablefmt='pretty'))
# print(x.info())
# print(x['bdate'].value_counts())
y = x['result']
x = x.drop('result', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

for i in range(1):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classfier = KNeighborsClassifier(n_neighbors=9)
    classfier.fit(x_train, y_train)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = classfier.predict(x_test)


print(accuracy_score(y_test,y_pred)*100)