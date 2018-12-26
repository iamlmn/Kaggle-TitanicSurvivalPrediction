import pandas as pd
from pandas import Series,DataFrame
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
import sklearn
from functools import reduce
train_df=pd.read_csv('train.csv')
dictsur=train_df.set_index('PassengerId')['Survived'].to_dict()
combined=pd.read_csv('combined.csv')

#Data preprocessing
def fill_age(row):
    grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.median()
    grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']] 
    condition = ((grouped_median_train['Sex'] == row['Sex']) &(grouped_median_train['Title'] == row['Title']) & (grouped_median_train['Pclass'] == row['Pclass'])) 
    return grouped_median_train[condition]['Age'].values[0]
def mfchild(passenger):
    age=passenger
    if age <16 :
        return 1
    else:
        return 0

def age_process():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    #status('age')
    return combined

def title_guess():
    grouped_train = combined.groupby(['Age','Sex','Alone'])
    grouped_median_train = grouped_train.median()
    grouped_median_train = grouped_median_train.reset_index()[['Age','Sex','Alone','Title']] 
    condition = ((grouped_median_train['Age'] == row['Age']) &(grouped_median_train['Sex'] == row['Sex']) & (grouped_median_train['Alone'] == row['Alone'])) 
    return grouped_median_train[condition]['Title'].values[0]

def get_title():
    global combined
    Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
    }
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    combined['Title'] = combined.Title.map(Title_Dictionary)
    
    #combined['Title'] = combined.apply(lambda row: title_guess(row) if row['Title']=='' else row['Title'], axis=1)
    return combined
def dummy_title():
    global combined
    title_dummies=pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined, title_dummies], axis=1)
    combined.drop('Title', axis=1, inplace=True)
    combined.drop('Title_Royalty', axis=1, inplace=True)
    return combined


def process_Sex():
    global combined
    combined['Sex']=combined['Sex'].map({'male':1,'female':0})
    return combined
def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    #status('fare')
    return combined
def process_embarked():
    global combined
    combined.Embarked.fillna('S', inplace=True)
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    #combined.drop('Embarked', axis=1, inplace=True)
    return combined

def process_pclass():
    global combined
    combined.Pclass.fillna('3', inplace=True)
    Pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')
    combined = pd.concat([combined, Pclass_dummies], axis=1)
    combined.drop('Pclass', axis=1, inplace=True)
    return combined
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
def Alone(passenger):
    sib,parch=passenger
    x=sib+parch
    if x==0:
        return 0
    else:
        return 1
def Bigfamily(passenger):
    sib,parch=passenger
    x=sib+parch
    if x >=4:
        return 1
    else:
        return 0
#def process_cabin():
def process_cabin():
    global combined    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin',axis=1, inplace=True)
    combined.drop('Cabin_U',axis=1, inplace=True)
    #status('cabin')
    return combined
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        #print(ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    combined.drop('Ticket_XXX', inplace=True, axis=1)
    return combined

if __name__=='__main__':

    combined['Child']=combined['Age'].apply(mfchild)
    combined['Alone']=combined[['SibSp','Parch']].apply(Alone,axis=1)
    combined['Bigfamily']=combined[['SibSp','Parch']].apply(Bigfamily,axis=1)
    combined=get_title()
    combined=age_process()
    combined=process_embarked()
    combined=process_pclass()
    combined=process_Sex()
    combined=process_fares()
    combined=process_cabin()
    combined = process_ticket()
    combined.drop('PassengerId',axis=1, inplace=True)
    combined.drop('Name',axis=1, inplace=True)
    combined.drop('SibSp',axis=1, inplace=True)
    combined.drop('Parch',axis=1, inplace=True)
    #combined.drop('Ticket',axis=1, inplace=True)
    combined=dummy_title()
    combined.to_csv('output.csv',index=False)
    train=combined[:891]
    test=combined[891:]
    targets=train_df['Survived']

    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf=GradientBoostingClassifier()
    clf = clf.fit(train, targets)

    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    features.plot(kind='barh', figsize=(25, 25)).figure.savefig('Features check')
    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    test_reduced = model.transform(test)
    logreg = LogisticRegression()
    logreg_cv = LogisticRegressionCV()
    rf = RandomForestClassifier()
    gboost = GradientBoostingClassifier()
    dtc=DecisionTreeClassifier()

    models = [logreg, logreg_cv, rf, gboost,dtc]

    for model in models:
        print ('Cross-validation of : {0}'.format(model.__class__))
        score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
        print ('CV score = {0}'.format(score))
        print ('****')
    output = clf.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv('Submitting_output.csv', index=False)
