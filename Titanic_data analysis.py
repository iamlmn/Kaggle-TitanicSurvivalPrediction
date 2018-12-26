import pandas as pd
from pandas import Series,DataFrame
titanic_df=pd.read_csv('train.csv')
print(titanic_df.head())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.factorplot('Sex',data=titanic_df,kind='count').savefig('Sex count plot')
sns.factorplot('Pclass',data=titanic_df,kind='count',hue='Sex').savefig('Pclass vs Sex')

def mfchild(passenger):
    age,sex=passenger
    if age <16:
        return 'child'
    else:
        return sex
titanic_df['person']=titanic_df[['Age','Sex']].apply(mfchild,axis=1)
print(titanic_df.head(10))
#titanic_df['person'].hist().figure.savefig('hist of persons')
sns.factorplot('Pclass',data=titanic_df,kind='count',hue='person').savefig('Pclass vs person')


#KDE plot using SEX
fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig.savefig('facetgrid plot for sex')

#KDE plot using person
fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig.savefig('facetgrid plot for person')

#KDE plot using Pclass
fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig.savefig('facetgrid plot for person')

#Dropping cabin with na values
deck=titanic_df['Cabin'].dropna()
levels=[]
for level in deck:
    levels.append(level[0])
cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']

sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count').savefig('Cabin count')


#Without Cabin T cuz it doesnt make sense
cabin_df=cabin_df[cabin_df['Cabin']!='T']
sns.factorplot('Cabin',data=cabin_df,palette='summer',kind='count').savefig('Cabin count without T')

#Embarkation plot based on class
sns.factorplot('Embarked',data=titanic_df,palette='winter_d',kind='count',hue='Pclass').savefig('Emarkation based on class')

#Who all are alone?
FA=DataFrame()
'''FA=titanic_df[titanic_df['Parch']==0]
FA=FA[FA['SibSp']==0]'''
def FAFAM(passenger):
    sib,par=passenger
    if sib!=0 or par!=0:
        return 'Family'
    else:
        return 'Alone'
titanic_df['Famoralone']=titanic_df[['SibSp','Parch']].apply(FAFAM,axis=1)
sns.factorplot('Famoralone',data=titanic_df,kind='count').savefig('Family or alone')
sns.factorplot('Famoralone',data=titanic_df,hue='Sex',kind='count').savefig('Family or alone sex wise')
sns.factorplot('Famoralone',data=titanic_df,hue='Pclass',kind='count',palette='Blues').savefig('Family or alone class wise')


#Survival observations

titanic_df['Survivor']=titanic_df.Survived.map({0:'no',1:'Yes'})
#How many survived vs dead
sns.factorplot('Survivor',data=titanic_df,kind='count',palette='Set1').savefig('DeadVSalive')
#Attributes to death
sns.factorplot('Survivor',data=titanic_df,kind='count',palette='Set1',hue='Pclass').savefig('DeadVSalive due to pclass')
sns.factorplot('Pclass','Survived',data=titanic_df).savefig('DeadVSalive due to pclass')
sns.factorplot('Pclass','Survived',data=titanic_df,hue='person').savefig('DeadVSalive due topclass based on person')

