def random_imputator(data,fea):
    data[fea+'_filled']=data[fea].copy()
    s=data.dropna().sample(data[fea].isnull().sum(),random_state=42).index
    s.index=data[data[fea].isnull()].index
    data.loc[data[fea].isnull(),fea+'_filled']=s
