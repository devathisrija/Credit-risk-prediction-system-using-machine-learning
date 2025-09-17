import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
import handling_outliers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import missing_values
import variable_transformation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from logging_file import setup_logging
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
logger=setup_logging("main code file")
class FIRST:
    def __init__(self,df):
        try:
            self.df=pd.read_csv(df)
            logger.info(self.df.isnull().sum())
            logger.info(self.df.tail(2))
            self.df=self.df.drop([150000,150001],axis=0)
            self.df=self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(self.df.isnull().sum())
            self.df['NumberOfDependents'] = pd.to_numeric(self.df['NumberOfDependents'])
            self.x=self.df.drop(['Good_Bad'],axis=1)
            self.y=self.df.iloc[:,-1]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=42)
            logger.info(self.x_train.shape)
            logger.info(self.x_test.shape)
            logger.info(self.df['NumberOfDependents'].dtype)

        except Exception as e:
            logger.info(e)


    def handle_missing_values(self):
        try:
            missing_values.random_imputator(self.x_train,'MonthlyIncome')
            missing_values.random_imputator(self.x_train,'NumberOfDependents')
            logger.info(self.x_train.isnull().sum())
            logger.info(self.x_train['MonthlyIncome'].std())
            logger.info(self.x_train['MonthlyIncome_filled'].std())
            missing_values.random_imputator(self.x_test, 'MonthlyIncome')
            missing_values.random_imputator(self.x_test, 'NumberOfDependents')
            logger.info(self.x_test.isnull().sum())
            self.x_train=self.x_train.drop(['MonthlyIncome','NumberOfDependents'],axis=1)
            self.x_test=self.x_test.drop(['MonthlyIncome','NumberOfDependents'],axis=1)
            logger.info(self.x_train.columns)
            logger.info(self.x_test.columns)
            logger.info(self.x_train.isnull().sum())
            logger.info(self.x_test.isnull().sum())

        except Exception as e:
            logger.info(e)

    def variable_transform(self):
        try:
            self.x_train_num=self.x_train.select_dtypes(exclude=object)
            self.x_train_cat=self.x_train.select_dtypes(include=object)
            self.x_test_num = self.x_test.select_dtypes(exclude=object)
            self.x_test_cat = self.x_test.select_dtypes(include=object)
            variable_transformation.transform(self.x_train_num)
            variable_transformation.transform(self.x_test_num)
            logger.info(self.x_train.columns)
            l=[]
            for i in self.x_train_num.columns:
                if "_sqrt" not in i:
                    l.append(i)
            self.x_train_num=self.x_train_num.drop(l,axis=1)
            self.x_test_num=self.x_test_num.drop(l,axis=1)
            logger.info(self.x_train_num.columns)
            logger.info(self.x_train_num.shape)
        except Exception as e:
            logger.info(e)

    def outliers(self):
        try:
            for i in self.x_train_num.columns:
                ll,ul=handling_outliers.trimming(self.x_train_num,i)
                self.x_train_num[i+'_trim']=np.where(self.x_train_num[i]>ul,ul,
                                             np.where(self.x_train_num[i]<ll,ll,self.x_train_num[i]))

            for i in self.x_test_num.columns:
                ll, ul = handling_outliers.trimming(self.x_test_num, i)
                self.x_test_num[i+'_trim'] = np.where(self.x_test_num[i] > ul, ul,
                                       np.where(self.x_test_num[i] < ll, ll, self.x_test_num[i]))
            l=[]
            for i in self.x_train_num.columns:
                if "_trim" not in i:
                    l.append(i)
            self.x_train_num=self.x_train_num.drop(l,axis=1)
            self.x_test_num=self.x_test_num.drop(l,axis=1)
            logger.info(self.x_train_num.columns)
            logger.info(self.x_train_num.shape)
            logger.info(f"outliers :{self.x_test_num.shape}")
        except Exception as e:
            logger.info(e)

    def cat_num(self):
        try:
            ohe=OneHotEncoder()
            res=ohe.fit_transform(self.x_train_cat[['Gender','Region']]).toarray()
            s=pd.DataFrame(res,columns=ohe.get_feature_names_out())
            logger.info(s)
            self.x_train_cat.reset_index(drop=True,inplace=True)
            self.x_train_cat=pd.concat([self.x_train_cat,s],axis=1)
            logger.info(self.x_train_cat.isnull().sum())
            self.x_train_cat=self.x_train_cat.drop(['Gender','Region'],axis=1)

            oe=OrdinalEncoder()
            res1=oe.fit_transform(self.x_train_cat[['Rented_OwnHouse','Occupation','Education']])
            s1=pd.DataFrame(res1,columns=oe.get_feature_names_out())
            self.x_train_cat=self.x_train_cat.drop(['Rented_OwnHouse','Occupation','Education'],axis=1)
            self.x_train_cat=pd.concat([self.x_train_cat,s1],axis=1)
            logger.info(self.x_train_cat.columns)

            ohe1 = OneHotEncoder()
            res2 = ohe1.fit_transform(self.x_test_cat[['Gender', 'Region']]).toarray()
            s3 = pd.DataFrame(res2, columns=ohe1.get_feature_names_out())
            logger.info(s)
            self.x_test_cat.reset_index(drop=True, inplace=True)
            self.x_test_cat = pd.concat([self.x_test_cat, s3], axis=1)
            logger.info(self.x_test_cat.isnull().sum())
            self.x_test_cat = self.x_test_cat.drop(['Gender', 'Region'], axis=1)

            oe1 = OrdinalEncoder()
            res3 = oe1.fit_transform(self.x_test_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            s4 = pd.DataFrame(res3, columns=oe1.get_feature_names_out())
            self.x_test_cat = self.x_test_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)
            self.x_test_cat = pd.concat([self.x_test_cat, s4], axis=1)
            logger.info(f"x_test : {self.x_test_cat.columns}")

        except Exception as e:
            logger.info(e)


    def filter(self):
        try:
            logger.info(f"before: {self.x_train_num.shape}")
            logger.info(self.x_test_num.shape)

            vt=VarianceThreshold()
            vt.fit(self.x_train_num)
            logger.info(sum(vt.get_support()))
            logger.info(self.x_train_num.columns[vt.get_support()])
            constant = self.x_train_num.columns[~vt.get_support()]

            '''vt1=VarianceThreshold(threshold=0.1)
            vt1.fit(self.x_train_num)
            logger.info(sum(vt1.get_support()))
            logger.info(self.x_train_num.columns[vt1.get_support()])
            constant1 = self.x_train_num.columns[~vt1.get_support()]'''''
            logger.info(f"constant :{constant}")
            self.x_train_num=self.x_train_num.drop(constant,axis=1)
            self.x_test_num=self.x_test_num.drop(constant,axis=1)
            logger.info(self.x_train_num.shape)
            logger.info(self.x_test_num.shape)

        except Exception as e:
            logger.info(e)

    def convert(self):
        try:
            ob=LabelEncoder()
            self.y_train=ob.fit_transform(self.y_train)
            self.y_test=ob.fit_transform(self.y_test)
        except Exception as e:
            logger.info(e)


    def correlation(self):
        try:
            for i in self.x_train_num.columns:
                if pearsonr(self.x_train_num[i],self.y_train)[1]>0.05:
                    self.x_train_num=self.x_train_num.drop(i,axis=1)
                    self.x_test_num=self.x_test_num.drop(i,axis=1)
            logger.info(f"correlation :{self.x_train_num.columns}")
        except Exception as e:
            logger.info(e)

    def combine(self):
        try:
            self.x_train_num=self.x_train_num.reset_index(drop=True)
            self.training_data=self.x_train_num.copy()
            self.training_data=pd.concat([self.training_data,self.x_train_cat],axis=1)
            logger.info(f"training data shape :{self.training_data.shape}")
            self.x_test_num=self.x_test_num.reset_index(drop=True)
            self.testing_data=self.x_test_num.copy()
            self.testing_data=pd.concat([self.testing_data,self.x_test_cat],axis=1)
            logger.info(f"testing data shape :{self.testing_data.shape}")
        except Exception as e:
            logger.info(e)

    def scaling(self):
        try:
            sc=StandardScaler()
            self.training_data_scaled=sc.fit_transform(self.training_data)
            self.testing_data_scaled=sc.fit_transform(self.testing_data)
        except Exception as e:
            logger.info(e)

    def balancing(self):
        try:
            logger.info(f"before y_train->0 {sum(self.y_train==0)}")
            logger.info(f"before y_train->1 {sum(self.y_train == 1)}")
            sm=SMOTE(random_state=42)
            self.training_data_balanced,self.y_train_balanced=sm.fit_resample(self.training_data_scaled,self.y_train)
            logger.info(f"after y_train->0 {sum(self.y_train_balanced == 0)}")
            logger.info(f"after y_train->1 {sum(self.y_train_balanced == 1)}")

        except Exception as e:
            logger.info(e)

    def model_selection(self):
        try:
            self.lr=LogisticRegression()
            self.nb=GaussianNB()
            self.dt=DecisionTreeClassifier(criterion='entropy')
            self.rf=RandomForestClassifier(n_estimators=5,criterion='entropy')
            self.knn=KNeighborsClassifier(n_neighbors=5)

            self.lr.fit(self.training_data_balanced,self.y_train_balanced)
            self.nb.fit(self.training_data_balanced, self.y_train_balanced)
            self.dt.fit(self.training_data_balanced, self.y_train_balanced)
            self.rf.fit(self.training_data_balanced, self.y_train_balanced)
            self.knn.fit(self.training_data_balanced, self.y_train_balanced)

            lrp=self.lr.predict(self.training_data_balanced)
            nbp = self.nb.predict(self.training_data_balanced)
            dtp = self.dt.predict(self.training_data_balanced)
            rfp=self.rf.predict(self.training_data_balanced)

            lrfpr,lrtpr,lrthr=roc_curve(self.y_train_balanced,lrp)
            nbfpr, nbtpr, nbthr = roc_curve(self.y_train_balanced, nbp)
            dtfpr, dttpr, dtthr = roc_curve(self.y_train_balanced, dtp)
            rffpr, rftpr, rfthr = roc_curve(self.y_train_balanced, rfp)

            '''plt.figure(figsize=(5, 3))
            plt.plot([0, 1], [0, 1], "k--", label='50% AUC')
            plt.plot(lrfpr, lrtpr, label='lr')
            plt.plot(nbfpr, nbtpr, label='nb')
            plt.plot(dtfpr, dttpr, label='dt')
            plt.plot(rffpr, rftpr, label='rf')
            plt.legend()

            plt.show()'''
        except Exception as e:
            logger.info(e)

    def hyper_parameter_tuning(self):
        try:
            parameters = {

                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None]
            }
            self.reg=DecisionTreeClassifier()
            gsc=GridSearchCV(estimator=self.reg,param_grid=parameters,scoring='accuracy',cv=10)
            gsc1=gsc.fit(self.training_data_balanced,self.y_train_balanced)
            logger.info(gsc1.best_params_)
            logger.info(self.training_data.columns)
        except Exception as e:
            logger.info(e)
    def fit_model(self):
        try:
            self.dto=DecisionTreeClassifier(class_weight='balanced', criterion = 'entropy')
            self.dto.fit(self.training_data_balanced,self.y_train_balanced)
            with open('Credit_card_model_dt', 'wb') as a:
                pickle.dump(self.dto, a)
            with open('Credit_card_model_lr', 'wb') as b:
                pickle.dump(self.lr, b)
            with open('Credit_card_model_rf', 'wb') as c:
                pickle.dump(self.rf, c)
            with open('Credit_card_model_nb', 'wb') as d:
                pickle.dump(self.nb, d)
            with open('Credit_card_model_knn', 'wb') as g:
                pickle.dump(self.knn, g)
            logger.info("models dumped")
            logger.info(self.dto.predict([[0.2,28,4,5,1200,1,0,1,0,0,0,1,0,0,2,2]]))
        except Exception as e:
            logger.info(e)


if __name__=='__main__':
    try:
        data_path="C:\\Users\\srija\\Downloads\\creditcard.csv"
        obj=FIRST(data_path)
        obj.handle_missing_values()
        obj.variable_transform()
        obj.outliers()
        obj.cat_num()
        obj.filter()
        obj.convert()
        obj.correlation()
        obj.combine()
        obj.scaling()
        obj.balancing()
        obj.model_selection()
        obj.hyper_parameter_tuning()
        obj.fit_model()
    except Exception as e:
        logger.info(e)