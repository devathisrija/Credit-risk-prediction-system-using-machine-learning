from sklearn.preprocessing import OneHotEncoder
def nominal(data,l):
    o=OneHotEncoder()
    o.fit(data[l])
