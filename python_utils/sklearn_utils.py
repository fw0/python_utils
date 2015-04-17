import pandas as pd
import itertools
from sklearn.base import TransformerMixin as TransformerMixin
import numpy as np

class myTransformerMixin(TransformerMixin):

    def fit(self, data):
        pass
    
    def fit_transform(self):
        pass

class transform_from_fxn(myTransformerMixin):

    def __init__(self, horse):
        self.horse = horse

    def transform(self, *args, **kwargs):
        return self.horse(*args, **kwargs)

    def fit(self):
        pass


class pd_filter_transform(myTransformerMixin):

    def __init__(self, bool_f):
        self.bool_f = bool_f

    def transform(self, x):
        return x.loc[x.apply(self.bool_f,axis=1)]

class pd_filter_index_transform(myTransformerMixin):

    def __init__(self, bool_f):
        self.bool_f = bool_f

    def transform(self, x):
        return x.loc[map(self.bool_f, x.index.values)]
    

class pd_join_transform(myTransformerMixin):

    def transform(self, *args):
        return tuple([y.loc[reduce(pd.Index.intersection, [x.index for x in args])] for y in args])


class zip_transform(myTransformerMixin):

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, *args):
        return tuple([transform.transform(x) for (transform,x) in zip(self.transforms, args)])

        
    
def pd_train_test_split(X, y=None, test_size=0.5, random_state=0):

    np.random.seed(random_state)
    N = X.shape[0]
    in_train = np.array([np.random.uniform() < test_size for i in xrange(N)], dtype=bool)
    in_test = np.invert(in_train)
    if not y is None:
        return X.iloc[in_train,:], X.iloc[in_test,:], y.iloc[in_train], y.iloc[in_test]
    else:
        return X.iloc[in_train,:], X.iloc[in_test,:]

def pd_predict_wrapper(predict_f):
        
    def wrapped_f(X_test):
        if isinstance(X_test, pd.DataFrame):
            test_index = X_test.index

            def convert(x):
                if len(x.shape) == 1:
                    return pd.Series(x, index=test_index)
                elif len(x.shape) == 2:
                    return pd.DataFrame(x, index=test_index)
            
            ans = predict_f(X_test.as_matrix())

            if isinstance(ans, tuple):
                return tuple(map(convert,ans))
            else:
                return convert(ans)
        else:
            return predict_f(X_test)

    return wrapped_f
    
    
class index_optional_fitter_wrapper(object):
    """
    assume that if input has index, then predict's output also has index
    """
    def __init__(self, horse, flag):
        self.horse, self.flag = horse, flag

    def fit(self, *args, **kwargs):

        def convert(v, flag):
            if flag in ['df','series']:
                return v.as_matrix()
            elif flag == None:
                return v
            assert False

        return index_optional_predictor_wrapper(self.horse.fit(*itertools.starmap(convert, zip(args, self.flags))))


class index_optional_predictor_wrapper(object):

    def __init__(self, horse):
        self.horse = horse
    
    def predict(self, X):
        y = self.horse.predict(X)
        if isinstance(y, pd.Series):
            return pd.Series(y, index=X.index)
        elif isinstance(y, pd.DataFrame):
            return pd.DataFrame(y, index=X.index)
        else:
            return y


class supervised_filter_transformer(TransformerMixin):

    def __init__(self, predictor, top_k):
        self.predictor, self.top_k = predictor, top_k

    def transform(self, X):
        pass
    

class supervised_filter_fitter(TransformerMixin):

    def __init__(self, fitter, top_k):
        self.fitter, self.top_k = fitter, top_k

    def fit(self, X, y):
        predictor = self.fitter.fit(X, y)
