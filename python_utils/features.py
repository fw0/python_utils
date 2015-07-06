import itertools
import pandas as pd
from sklearn.base import TransformerMixin as TransformerMixin
import pdb
import numpy as np
import string

class bin(object):

    def __contains__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    
class equals_bin(bin):

    def __init__(self, val):
        self.val = val

    def __contains__(self, x):
        return x == self.val

    def __repr__(self):
        return 'equals_%s' % repr(self.val)

    
class contains_bin(bin):

    def __init__(self, vals):
        self.vals = vals

    def __contains__(self, x):
        return x in self.vals

    def __repr__(self):
        return repr(self.vals)


class interval_bin(bin):
    """
    return x \in [low,high)
    """
    def __init__(self, low, high):
        self.low, self.high = low, high

    def __contains__(self, x):
        return (self.low is None or self.low <= x) and (self.high is None or x < self.high)

    def __repr__(self):
        return 'bin_%s_%s' % (repr(self.low), repr(self.high))


class not_bin(bin):

    def __init__(self, bin):
        self.bin = bin

    def __contains__(self, x):
        return not x in self.bin

    def __repr__(self):
        return 'not_%s' % repr(self.bin)
    

class union_bin(bin):

    def __init__(self, bins):
        self.bins = bins

    def __contains__(self, x):
        for bin in self.bins:
            if x in bin:
                return True
        return False

    def __repr__(self):
        return 'union_%s' % string.join([repr(bin) for bin in self.bins], sep='_')
        

class feature(object):
    """
    is just any function that has repr
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class map_feature(feature):
    """
    accepts dictionary of bin->val
    """
    def __init__(self, bin_d):
        self.bin_d = bin_d

    def __call__(self, x):
        try:
            for (bin,val) in self.bin_d.iteritems():
                if x in bin:
                    return val
        except:
            print self.bin_d
            pdb.set_trace()
        return np.nan

    @classmethod
    def from_map_d(cls, map_d):
        d = {}
        for (key,val) in map_d.iteritems():
            d[equals_bin(key)] = val
        return map_feature(d)
        
        
class df_feature(feature):
    """
    features that accept a dataframe.  returns a series or dataframe
    """    
    pass


class empirical_cat_df_feature(feature):
    """
    categorical binary df_feature that determines the levels from what is actually present
    """
    def __init__(self, col_name):
        self.col_name = col_name
    
    def __call__(self, df):
        levels = df[self.col_name].unique()
        bin_feat = bins_feature([equals_bin(level) for level in levels])
        df_feature = df_feature_from_feature(bin_feat, self.col_name)
        return df_feature(df)
    

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
        #ans.columns = [self.col_name]
        if isinstance(ans, pd.DataFrame):
            ans.columns = ['%s_%s' % (self.col_name,col) for col in ans.columns]
        else:
            ans = pd.DataFrame({self.col_name:ans})
        #print ans
        #ans.columns = ['%s' % col for col in ans.columns]
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
    """
    if not_others is true, adds a bin that is true if none of the other bins return True
    """
    def __init__(self, bins, not_others=False):
        try:
            iter(bins)
        except TypeError:
            self.bins = [bins]
        else:
            self.bins = bins
        self.not_others = not_others
    
    def __call__(self, x):
        ans = [float(x in bin) for bin in self.bins]
        index = [repr(bin) for bin in self.bins]
        if self.not_others:
            ans += [float(np.sum(ans) < .001)]
            index += ['none_of']
        return pd.Series(ans, index=index)
        return pd.Series([float(x in bin) for bin in self.bins], index = [repr(bin) for bin in self.bins])

    def __repr__(self):
        import string
        return string.join([repr(bin for bin in self.bins)], sep='_')

    @classmethod
    def from_boundaries(cls, boundaries, drop=True):
        """
        doesn't make bin for last window
        """
        if drop:
            return bins_feature([interval_bin(low,high) for (low,high) in itertools.izip(boundaries[0:-2],boundaries[1:-1])])
        else:
            return bins_feature([interval_bin(low,high) for (low,high) in itertools.izip(boundaries[0:-1],boundaries[1:len(boundaries)])])


