import os
#import cPickle as pickle
#import marshal as pickle
import dill as pickle
import hashlib
import functools
import pdb
import inspect
import numpy as np
import pandas as pd
import python_utils.python_utils.decorators as decorators
import time
import python_utils.python_utils.basic as basic_utils


"""
before any functions are used, first have to set the constants module
"""
cache_folder = None
which_hash_f = None
cache_max_size = None
fig_archiver = None

def init(_cache_folder, _which_hash_f, _fig_archiver, _cache_max_size = 999999):
    global cache_folder
    global which_hash_f
    global cache_max_size
    global fig_archiver
    cache_folder = _cache_folder
    which_hash_f = _which_hash_f
    cache_max_size = _cache_max_size
    fig_archiver = _fig_archiver

#@timeit_fxn_decorator
def get_hash(obj):
    try:
        pickle_s = pickle.dumps(obj)
    except TypeError as e:
        print e
        pdb.set_trace()
    m = hashlib.new(which_hash_f)
    m.update(pickle_s)
    return m.hexdigest()


def generic_get_arg_key(*args, **kwargs):
    return get_hash((args, kwargs))


def generic_get_key(identifier, *args, **kwargs):
    ans = '%s%s' % (get_hash(identifier), get_hash((args, kwargs)))
#    print args, ans
    return ans


def id_get_key(identifier, *args, **kwargs):
    ans = str(map(id, args))
#    print ans
    return ans
#    return '%s%s' % (get_hash(identifier), get_hash((args, kwargs)))

def hash_get_key(identifier, *args, **kwargs):
#    return hash(sum(map(hash, args + kwargs.values())))

    try:
        return hash(hash(identifier)+sum(map(hash, args)))
    except:
        import pdb
        pdb.set_trace()

def generic_get_path(identifier, *args, **kwargs):
    import types
    import pdb
    def get_identifier_folder(iden):
        if isinstance(iden, str):
            return iden
        if isinstance(iden, functools.partial):
            return get_identifier_folder(iden.func)
        elif isinstance(iden, types.FunctionType):
            assert False
            return iden.__name__
        else:
            return iden.__class__.__name__
    return '%s/%s/%s' % (cache_folder, get_identifier_folder(identifier), generic_get_key(identifier, *args, **kwargs))


def get_temp_path(obj):
    return '%s/%s' % (cache_folder, get_hash(obj))


def read_pickle(file_path):
    beg = time.time()
    try:
        ans = pickle.load(open(file_path, 'rb'))
    except:
        print file_path
        pdb.set_trace()
    return ans


def read(f, read_f, path_f, identifier, file_suffix, *args, **kwargs):
    file_path = '%s.%s' % (path_f(identifier, *args, **kwargs), file_suffix)
    if os.path.exists(file_path):
#        print identifier, file_path, 'SUCCESS'
        return read_f(file_path)
    else:
#        print identifier, file_path, 'NOT FOUND'
        return f(*args, **kwargs)


class read_fxn_decorator(decorators.fxn_decorator):
    
    def __init__(self, read_f, path_f, file_suffix):
        self.read_f, self.path_f, self.file_suffix = read_f, path_f, file_suffix

    def __call__(self, f):

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            print basic_utils.get_callable_name(f), 'gggggg'
            return read(f, self.read_f, self.path_f, basic_utils.get_callable_name(f), self.file_suffix, *args, **kwargs)

        return wrapped_f


class read_decorated_method(decorators.decorated_method):

    def __init__(self, f, read_f, path_f, file_suffix):
        self.f, self.read_f, self.path_f, self.file_suffix = f, read_f, path_f, file_suffix

#    @timeit_method_decorator()
    def __call__(self, inst, *args, **kwargs):
        return read(functools.partial(self.f, inst), self.read_f, self.path_f, inst, self.file_suffix, *args, **kwargs)


class read_method_decorator(decorators.method_decorator):

    def __init__(self, read_f, path_f, file_suffix):
        self.read_f, self.path_f, self.file_suffix = read_f, path_f, file_suffix

    def __call__(self, f):
        return read_decorated_method(f, self.read_f, self.path_f, self.file_suffix)


def write_pickle(obj, file_path):
    f = open(file_path, 'wb')
    pickle.dump(obj, f)
    f.close()


def write(f, write_f, path_f, identifier, file_suffix, *args, **kwargs):
    """
    only write if ans is not null
    path_f(identifier, *args, **kwargs)
    write_f(ans, full_file_path)
    file_suffix can be None
    """
    if file_suffix == None:
        file_path = path_f(identifier, *args, **kwargs)
    else:
        file_path = '%s.%s' % (path_f(identifier, *args, **kwargs), file_suffix)
    ans = f(*args, **kwargs)
    #print 'write', identifier, file_path, 
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    write_f(ans, file_path)

    return ans


class write_fxn_decorator(decorators.fxn_decorator):

    def __init__(self, write_f, path_f, file_suffix):
        self.write_f, self.path_f, self.file_suffix = write_f, path_f, file_suffix

    def __call__(self, f):

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return write(f, self.write_f, self.path_f, basic_utils.get_callable_name(f), self.file_suffix, *args, **kwargs)

        return wrapped_f


class write_decorated_method(decorators.decorated_method):

    def __init__(self, f, write_f, path_f, file_suffix):
        self.f, self.write_f, self.path_f, self.file_suffix = f, write_f, path_f, file_suffix

    def __call__(self, inst, *args, **kwargs):
        # FIX: inst should actually be self.f
        return write(functools.partial(self.f, inst), self.write_f, self.path_f, inst, self.file_suffix, *args, **kwargs)


class write_method_decorator(decorators.method_decorator):

    def __init__(self, write_f, path_f, file_suffix):
        self.write_f, self.path_f, self.file_suffix = write_f, path_f, file_suffix

    def __call__(self, f):
        """
        this call performs the act of replacing the existing method
        """
        return write_decorated_method(f, self.write_f, self.path_f, self.file_suffix)

def cache(f, key_f, identifier, d, *args, **kwargs):
    #return f(*args, **kwargs)
    #print 'D before', [(key,id(val),id(val[0]),val) for (key,val) in d.iteritems()]
#    pdb.set_trace()
    if len(d) > cache_max_size:
        d.clear()
#    pdb.set_trace()
    key = key_f(identifier, *args, **kwargs)
#    return f(*args, **kwargs)
    try:
#        print 'good'#, key
        ans = d[key]
#        print 'compute OLD', f, args, kwargs
    except KeyError:
#        print 'gg'#, key
#        print 'compute NEW', f, args, kwargs
#        pdb.set_trace()
        ans = f(*args, **kwargs)
        d[key] = ans
    #print 'AFTER', id(d), [(key,id(val),id(val[0]),val) for (key,val) in d.iteritems()], args, key, ans
    #print 'compare', f(*args, **kwargs), args[0], ans
    #assert np.sum(f(*args, **kwargs) - args[0]) < .0001
    return ans


class cache_fxn_decorator(decorators.fxn_decorator):

    def __init__(self, key_f):
        self.key_f = key_f
        self.d = {}
        print key_f, 'key_f'

    def __call__(self, f):
        print 'wrapping', f
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return cache(f, self.key_f, basic_utils.get_callable_name(f), self.d, *args, **kwargs)

        return wrapped_f

class cache_decorated_method(decorators.decorated_method):

    def __init__(self, f, key_f, d):
        self.f, self.key_f, self.d = f, key_f, d

    def __call__(self, inst, *args, **kwargs):
        return cache(functools.partial(self.f, inst), self.key_f, inst, self.d, *args, **kwargs)

class cache_method_decorator(decorators.method_decorator):

    def __init__(self, key_f):
        self.key_f = key_f
        self.d = {}

    def __call__(self, f):
        return cache_decorated_method(f, self.key_f, self.d)

#import python_utils.utils as utils

default_read_method_decorator = lambda: read_method_decorator(read_pickle, generic_get_path, 'pickle')
default_write_method_decorator = lambda: write_method_decorator(write_pickle, generic_get_path, 'pickle')
default_cache_method_decorator = lambda: cache_method_decorator(generic_get_key)

#default_everything_method_decorator = utils.multiple_composed_f(default_cache_method_decorator, default_read_method_decorator, default_write_method_decorator)
"""
@caching.default_cache_method_decorator
@caching.default_read_method_decorator
@caching.default_write_method_decorator
"""

default_read_fxn_decorator = lambda: read_fxn_decorator(read_pickle, generic_get_path, 'pickle')
default_write_fxn_decorator = lambda: write_fxn_decorator(write_pickle, generic_get_path, 'pickle')
default_cache_fxn_decorator = lambda: cache_fxn_decorator(generic_get_key)
id_cache_fxn_decorator = lambda: cache_fxn_decorator(generic_get_key)
#default_everything_fxn_decorator = utils.multiple_composed_f(default_cache_fxn_decorator, default_read_fxn_decorator, default_write_fxn_decorator)
"""
@caching.default_cache_fxn_decorator
@caching.default_read_fxn_decorator
@caching.default_write_fxn_decorator
"""

hash_cache_method_decorator = lambda: cache_method_decorator(generic_get_key)
