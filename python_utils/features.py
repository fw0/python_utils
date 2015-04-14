import itertools
import pandas as pd
from sklearn.base import TransformerMixin as TransformerMixin
import pdb

class bin(object):

    def __contains__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class interval_bin(bin):

    def __init__(self, low, high):
        self.low, self.high = low, high

    def __contains__(self, x):
        return (self.low is None or self.low <= x) and (self.high is None or x < self.high)

    def __repr__(self):
        return 'bin_%s_%s' % (repr(self.low), repr(self.high))
    

class feature(object):
    """
    is just any function that has repr
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
    
class df_feature(feature):
    """
    features that accept a dataframe.  returns a series or dataframe
    """    
    pass


def df_from_df_and_df_features(df_features, df):
    vals = [df_feature(df) for df_feature in df_features]
    for (i,(val, df_feature)) in enumerate(itertools.izip(vals, df_features)):
        if isinstance(val, pd.Series):
            vals[i] = pd.DataFrame({repr(df_feature):val})
    return pd.concat(vals, axis=1)


class df_feature_from_feature(df_feature):
    """
    applies feature to a column of dataframe.  
    """
    def __init__(self, horse, col_name):
        self.horse, self.col_name = horse, col_name

    def __call__(self, df):
        ans = df[self.col_name].apply(self.horse)
        ans.columns = ['%s_%s' % (self.col_name,col) for col in ans.columns]
        return ans
        
    def __repr__(self):
        return '' % (self.col_name, repr(self.horse))
        
class bin_feature(feature):

    def __init__(self, bin):
        self.bin = bin
    
    def __call__(self, x):
        return pd.Series({repr(self):(x in self.bin)})

    def __repr__(self):
        return repr(self.bin)


class bins_feature(feature):

    def __init__(self, bins):
        try:
            iter(bins)
        except TypeError:
            self.bins = [bins]
        else:
            self.bins = bins
    
    def __call__(self, x):
        return pd.Series([float(x in bin) for bin in self.bins], index = [repr(bin) for bin in self.bins])

    def __repr__(self):
        import string
        return string.join([repr(bin for bin in self.bins)], sep='_')

    @classmethod
    def from_boundaries(cls, boundaries):
        """
        doesn't make bin for last window
        """
        return bins_feature([interval_bin(low,high) for (low,high) in itertools.izip(boundaries[0:-2],boundaries[1:-1])])


