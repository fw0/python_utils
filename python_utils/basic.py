import functools
import json
import types
import python_utils.python_utils.decorators as decorators
import numpy as np
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm
import time
import os
import imp
import pdb
import copy
import datetime, itertools
import pandas as pd

class my_object(object):

    def __deepcopy__(self):
        pass

def get_callable_name(f):
    if isinstance(f, functools.partial):
        return get_callable_name(f.func)
    elif isinstance(f, types.FunctionType):
        return f.__name__
    else:
        try:
            return f.__class__.__name
        except:
            return repr(f)


get_shallow_rep = get_callable_name

def get_for_json(o):
    """
    first see if a rich representation is possible
    """
    if isinstance(o, functools.partial):
        return [\
            get_for_json(o.func),\
                [get_for_json(a) for a in o.args],\
                {k: get_for_json(v) for (k, v) in o.keywords.iteritems() if k[0] != '_'}\
                ]
    if isinstance(o, list):
        return [get_for_json(item) for item in o]
    try:
        d = o.__dict__
    except AttributeError:
        return get_shallow_rep(o)
    else:
        return [\
            get_shallow_rep(o),\
                {k:get_for_json(v) for (k, v) in d.iteritems() if k[0] != '_'}\
                ]

def display_fig_inline(fig):
#    import matplotlib.pyplot as plt
#    plt.show(fig)
#    fig.show()
#    return fig
    #import StringIO
    from io import StringIO
    import io
    from IPython.display import display
    from IPython.display import Image
    #output = StringIO.StringIO()
    #output = StringIO()
    output = io.BytesIO()
    fig.savefig(output, format='png')
    img = Image(output.getvalue())
    display(img, raw=False)
    import matplotlib.pyplot as plt
#    import caching
#    caching.fig_archiver.archive_fig(fig)
    plt.close()
    import gc
    gc.collect()
    return fig

def is_iterable(obj):
    try:
        temp = iter(obj)
    except TypeError:
        return False
    else:
        return True

class fig_archiver(object):

    def __init__(self, folder):
        self.folder = folder
        self.counter = 0
        import shutil
        pdb.set_trace()
        shutil.rmtree(self.folder)


    def __call__(self, fig):
        import os, pdb
        try:
            os.makedirs(self.folder)
        except Exception as e:
            print(e)
        fig.savefig('%s/%d' % (self.folder, self.counter))
        self.counter += 1
        return fig


class archiver(object):
    
    def __init__(self, folder, time_subfolder=True):
        if time_subfolder:
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d-%H-%M-%S")
            folder = '%s/%s' % (folder, date_str)
        import os, pdb
        import shutil
        #try:
        #    shutil.rmtree(folder)
        #except:
        #    pass
        self.folder = folder
        self.counter = 0
        try:
            os.makedirs(self.folder)
        except Exception as e:
            print(e)
        self.f = open('%s/log.txt' % self.folder, 'a')
        self.f.write('LOG START AT ' + str(datetime.now()).split('.')[0])

    def archive_fig(self, fig, folder=None, name=None):
        if True:
            display_fig_inline(fig)
            if folder is None:
                _folder = self.folder
            else:
                _folder = '%s/%s' % (self.folder, folder)
            try:
                os.makedirs(_folder)
            except Exception as e:
                pass
            if name is None:
                _name = '%d_%s' % (self.counter, str(datetime.datetime.now()).split('.')[0])
            else:
                _name = '%d_%s_%s' % (self.counter, str(datetime.datetime.now()).split('.')[0], name)
#            pdb.set_trace()
            fig.savefig('%s/%s' % (_folder, _name))
            self.counter += 1
            return fig

    def archive_string(self, s):
        f = open('%s/%d.txt' % (self.folder, self.counter), 'w')
        f.write(s)
        self.counter += 1
        print(s)
        f.close()
        return s

    def log_text(self, *s, **kwargs):
        if True:
            if 'folder' in kwargs:
                path = kwargs['folder']
            else:
                path = None
            print(str(datetime.datetime.now()).split('.')[0], s)
            if path is None:
                _folder = self.folder
            else:
                _folder = '%s/%s' % (self.folder, path)
            try:
                os.makedirs(_folder)
            except:
                pass
            if 'name' in kwargs:
                path = '%s/%s' % (_folder, kwargs['name'])
            else:
                path = '%s/log.txt' % _folder
            f = open(path, 'a')
            if ('write_time' not in kwargs ) or ('write_time' in kwargs and kwargs['write_time']):
                f.write(str(datetime.datetime.now()).split('.')[0] + str(s)+'\n')
            else:
                f.write(str(s)+'\n')
            f.flush()
            f.close()

    def log_text_s(self, s):
        print(s)
        self.f.write(str(s)+'\n')
        self.f.flush()

    def fig_text(self, ss):
        if True:
            import matplotlib.pyplot as plt
            import string
            fig, ax = plt.subplots()
            self.log_text('fig_text #%d' % self.counter)
            for s in ss:
                self.log_text(s)
            ax.text(0.,0., string.join(map(str,ss), sep='\n'), multialignment='left', horizontalalignment='left', verticalalignment='bottom', wrap=True)
            fig.tight_layout()
            import caching
            caching.fig_archiver.archive_fig(fig)
#        display_fig_inline(fig)
        

#import copy_reg
import types
import multiprocessing
#import multiprocessing_on_dill as multiprocessing


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

#copy_reg.pickle(types.MethodType, _pickle_method)

def joblib_parallel_map(num_processes, verbose, f, iterable):

    import joblib
    return joblib.Parallel(n_jobs=num_processes, verbose=verbose)(joblib.delayed(f)(x) for x in iterable)
#    return joblib.Parallel(n_jobs=num_processes, verbose=verbose, backend='threading')(joblib.delayed(f)(x) for x in iterable)

        
def parallel_map(num_processes, f, iterable):
    """                                                                                                                                                                                                    
    make a                                                                                                                                                                                                 
    """
#    import pathos.multiprocessing as multiprocessing
#    import multiprocess as multiprocessing
    #print iterable
    print(f)
    if num_processes is None:
        return list(map(f, iterable))
#    from pathos.multiprocessing import ProcessPool, ThreadPool

#    from pathos.pp import ParallelPool
#    pool = ParallelPool(nodes=num_processes)
#    return pool.map(f, iterable)

#    pool = ProcessPool(nodes=num_processes)
#    pool = ThreadPool(num_processes)
#    res = pool.map(f, iterable)
#    return res



    results = multiprocessing.Manager().list()
    iterable_queue = multiprocessing.Queue()

    def worker(_iterable_queue, _f, results_queue):
        for arg in iter(_iterable_queue.get, None):
            results_queue.append(_f(arg))

    for x in iterable:
        iterable_queue.put(x)

    for i in range(num_processes):
        iterable_queue.put(None)

    workers = []

    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(iterable_queue, f, results))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    return [x for x in results]

    
def compose(f, g):

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h
        
    
def get_grid_fig_axes(n_rows, n_cols, n, figsize=None):
    import matplotlib.pyplot as plt
    per_fig = n_rows * n_cols
    num_pages = (int(n) / per_fig) + (1 if (n % per_fig) > 0 else 0)
    figs = []
    axes = []
    for i in xrange(num_pages):
        start = per_fig * i
        end = min(per_fig * (i+1), n)
        assert (end - start) > 0
        if not (figsize is None):
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        for k in range(start,end):
            ax = fig.add_subplot(n_rows, n_cols, (k % per_fig)+1)
            axes.append(ax)
        #fig.tight_layout()
        figs.append(fig)
    return figs, axes

class static_var_fxn_decorator(decorators.fxn_decorator):

    def __init__(self, var_name, value):
        self.var_name, self.value = var_name, value

    def __call__(self, f):
        setattr(f, self.var_name, self.value)


def set_legend_unique_labels(ax, prop, **legend_kwargs):
    import operator
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = zip(*(dict(zip(labels, handles)).iteritems()))
    hl = sorted(zip(unique_handles, unique_labels), key=operator.itemgetter(1))
    sorted_handles, sorted_labels = zip(*hl)
    ax.legend(sorted_handles, sorted_labels, prop=prop, **legend_kwargs)

class noCacheException(Exception):
    pass

class raise_exception_fxn_decorator(decorators.fxn_decorator):

    def __init__(self, active=True):
        self.active = active
    
    def __call__(self, f):
        
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            if self.active:
                raise noCacheException
            else:
                return f(*args, **kwargs)

        return wrapped_f
    
class raise_exception_method(decorators.decorated_method):

    def __call__(self, inst, *args, **kwargs):
        raise noCacheException

    
class raise_exception_method_decorator(decorators.method_decorator):

    def __call__(self, f):
        return raise_exception_method()

def linuxtime_to_datetime(linuxtime):
    import datetime
    return datetime.datetime.fromtimestamp(
        int("1284101485")
    ).strftime('%Y-%m-%d %H:%M:%S')

def vals_to_rgbas(vals, cmap=cm.cool, vmin=None, vmax=None):

    def get_elt_rank(l):
        ans = np.zeros(len(l))
        level = 0
        for (i,pos) in enumerate(np.argsort(l)):
            if i > 0:
                if ans[pos] > ans[pos-1]:
                    level += 1
            ans[pos] = i
        return np.array(ans)

    #vals = get_elt_rank(vals) / len(vals)
    c = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    #c.autoscale(vals)
    m = cm.ScalarMappable(norm=c, cmap=cmap)
    colors = map(tuple,m.to_rgba(vals))
    return colors


def timeit(msg):

    def timeit_horse(method):
        
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()

            print('%s, %4.4f' % (msg, te-ts))
#            print '%r %4.4f sec' % \
#                (method.__name__, te-ts), msg, 'gg'
            #import pdb
#            pdb.set_trace()
            return result

        return timed

    return timeit_horse

def print_decorator(msg):
    def wrapper(f):
        def wrapped(*args, **kwargs):
            def shape(x):
                try:
                    return x.shape
                except:
                    return 'no shape'
#            print(msg), map(shape, args)
            ans = f(*args, **kwargs)
#            print ans
            return ans
        return wrapped
    return wrapper

import cProfile, pstats
#import StringIO

def call_graph(func):

    def wrapped(*args, **kwargs):
        output="profile.png"
        statsFileName="stats.pstats"
        import cProfile 


def do_cprofile(sort_by):
#    import StringIO
    import io
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:


 #               s = StringIO.StringIO()
                s = io.StringIO()
                print(sort_by)
                import pdb
#                s2 = StringIO.StringIO()
                s2 = 'profile_output.txt'
#                pdb.set_trace()
                profile.dump_stats(s2)
#                from graphviz import Source
#                src = Source(s2)
#                src.render('graph.png', view=True)  
#                pdb.set_trace()
#                ps = pstats.Stats(profile, stream=s).strip_dirs().sort_stats(sort_by)
                ps = pstats.Stats(profile, stream=s).sort_stats(sort_by)
#                ps = pstats.Stats(profile, stream=s).print_callers('dot')
                ps.print_stats()
#                pdb.set_trace()
                print(s.getvalue())

#                pdb.set_trace()
                ps.print_callers('dot')
#                print s.getvalue()
#            profile.print_stats()
        return profiled_func
    return wrapper

def plot_bar_chart(ax, labels, values, offset = 0, width = 0.75, label = None, alpha = 0.5, color = 'red', label_fontsize=None):
    num = len(labels)
    ax.bar(np.arange(num)+offset, values, label = label, alpha = alpha, color = color, width = width)
    ax.set_xticks(np.arange(num) + 1.0/2)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim((0, num))
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=label_fontsize)

def edit_distance(target, source):
    """
    Computes the min edit distance from target to source. Figure 3.25
    """
    n = len(target)
    m = len(source)
    print(target, source)
    insertCost = lambda x:1
    deleteCost = lambda x:1
    substCost = lambda x,y:1
    distance = [[0 for i in range(m+1)] for j in range(n+1)]
    for i in range(1,n+1):
        distance[i][0] = distance[i-1][0] + insertCost(target[i-1])
    for j in range(1,m+1):
        distance[0][j] = distance[0][j-1] + deleteCost(source[j-1])
    for i in range(1,n+1):
        for j in range(1,m+1):
           distance[i][j] = min(distance[i-1][j]+1,distance[i][j-1]+1, distance[i-1][j-1]+substCost(source[j-1],target[i-1]))
    return distance[n][m]

def scatter(xs, ys, plot_dim=0, colors=None):
    import matplotlib.pyplot as plt
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots()
    plot_xs = xs if len(xs.shape)==1 else xs[:,plot_dim]
    if colors is None:
        ax.scatter(plot_xs, ys, s=5.,edgecolors='none')
    else:
        ax.scatter(plot_xs, ys, c=colors,s=5.,edgecolors='none')
    display_fig_inline(fig)
    
def scatter_3d(xs, ys, zs, colors=None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs,ys,zs,c=colors,s=5.,edgecolors='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    display_fig_inline(fig)

def relative_depth(path, child_path):
    depth = 0
    while True:
        if child_path == '':
            assert False
        if path == child_path:
            return depth
        head, tail = os.path.split(child_path)
        child_path = head
        depth += 1
    assert False


def parent_import(path, name):
#    print 'importing', path, name
    while True:
        if path == '/':
            assert False
        try:
            print('ok', path, '%s/%s.py' % (path, name))
            return imp.load_source(name, '%s/%s.py' % (path, name))
        except IOError:
            head, tail = os.path.split(path)
            path = head
    assert False


def parent_find(path, file_name):
    print('find', path)
    while True:
        if path == '/':
            assert False
        _path, child_paths, file_names = os.walk(path).next()
        if file_name in file_names:
            return '%s/%s' % (path, file_name)
        else:
            head, tail = os.path.split(path)
            path = head
            print('find', path, file_name)
    assert False


def crawl(paths, is_leaf, is_child, f=None, mapper=map):

    def horse(path):
        root_path = path
        d = {}
        queue = [path]
        leaf_paths = []
        while len(queue) != 0:
            path = queue.pop()
            if is_leaf(root_path, path):
                leaf_paths.append(path)
#            d[path] = f(path)
            else:
                _path, child_paths, file_names = os.walk(path).next()
                assert path == _path
                candidate_child_paths = ['%s/%s' % (path,child_path) for child_path in child_paths]
                queue += [child_path for child_path in candidate_child_paths if is_child(root_path, child_path)]
        return leaf_paths
    if not isinstance(paths, str):
        leaf_pathss = [horse(path) for path in paths]
        leaf_paths = [path for leaf_paths in leaf_pathss for path in leaf_paths]
    else:
        leaf_paths = horse(paths)

    if f is None:
        return leaf_paths
    else:
        items = mapper(f, leaf_paths)
        return dict(zip(leaf_paths, items))
#    horse = lambda path: (path, f(path))
#    horse.__dict__ = f.__dict__
#    items = mapper(horse, leaf_paths)
#    return dict(items)


class node(object):

    def __init__(self, attrs=None, children=None, parent=None):
        if attrs is None:
            attrs = {}
        if children is None:
            children = {}
        self.attrs, self.children, self.parent = attrs, children, parent

    def parent_edge_name(self):
        assert not (self.parent is None)
        for (edge_name, child) in self.parent.children.iteritems():
            if child == self:
                return edge
        assert False


def identical_tree(_node, children_list):
#    _node = copy.copy(_node)
    if len(children_list) == 0:
        return _node
    else:
#        pdb.set_trace()
        for (edge_name, child_attrs) in children_list[0].iteritems():
            child = node(attrs=child_attrs)
            _node.children[edge_name] = identical_tree(child, children_list[1:])
            print('_____')
            print(_node )
            print(edge_name)
            print(child_attrs)
            print(_node.children)
            print('=====')
#        print 'children:', _node.children
        return _node


def write_directory_tree(path, _node):
    if not os.path.exists(path):
        os.makedirs(path)
    for (attr_name, attr) in _node.attrs.iteritems():
        attr.to_file('%s/%s' % (path, attr_name))
    for (edge_name, child) in _node.children.iteritems():
        write_directory_tree('%s/%s' % (path, edge_name), child)


class writeable_object(object):

    def __init__(self, obj, display_name=None):
        self.obj, self.display_name = obj, display_name

    def to_file(self, path):
        import inspect, string
        f = open(path, 'w')
        lines = inspect.getsourcelines(self.obj)[0]
        if self.display_name is None:
            f.write(lines[0])
        else:
            kind, rest = string.split(lines[0], maxsplit=1)
            name, rest = string.split(rest, '(')
            to_write = '%s %s(%s' % (kind, self.display_name, rest)
            f.write(to_write)
        for line in lines[1:]:
            f.write(line)
        f.close()

class from_string_writeable_object(object):

    def __init__(self, s):
        self.s = s

    def to_file(self, path):
        f = open(path, 'w')
        f.write(self.s)
        f.close()

def basic_map(mode, profile='', splat=False, exception_to_ignore=None):
    from ipyparallel import Client
    if mode != 'serial':
        assert profile != ''
        rc = Client(profile=profile)
        #rc[:].use_dill()
        rc[:].use_cloudpickle()
    
    if mode == 'direct':
        dview = rc[:]
        mapper = dview.map_sync
    elif mode == 'balanced':
        lview = rc.load_balanced_view()
        mapper = lview.map_sync
    elif mode == 'serial':
        mapper = map

    if not (exception_to_ignore is None):
        import python_utils.python_utils.caching as caching
        mapper = caching.ignore_exception_map_wrapper(mapper, noCacheException)

    def splat_mapper(f, iterable):
        def dec_f(args):
            return f(*args)
        return mapper(dec_f, iterable)

    return mapper if (not splat) else splat_mapper
    
def cached_map(mode, profile, splat, compute, recompute, reader=None, writer=None, get_path=None, suffix=None):
        
        
    import python_utils.python_utils.caching as caching
    mapper = basic_map(mode, profile, splat)
    mapper = caching.ignore_exception_map_wrapper(mapper, noCacheException)
    
    def final(f, iterable):
        decorated = caching.switched_decorator(f, compute, recompute, reader, writer, get_path, suffix)
        return mapper(decorated, iterable)
    
    return final

def parent_import_wrapper(path, s):
    return getattr(parent_import(path, s), s)(path)

def get_child_paths(path):
    _path, child_paths, file_names = os.walk(path).next()
    candidate_child_paths = ['%s/%s' % (path,child_path) for child_path in child_paths]

    return candidate_child_paths
#    actual_child_paths = [child_path for child_path in candidate_child_paths if is_child(root_path, child_path)]

def hardcoded_crawl(paths, depth=None, f=None, mapper=map):

    def is_child(path, cur_path):
        head, tail = os.path.split(cur_path)
        return (not (tail[0] in ['_','.'])) and (tail != 'cache')

    def is_leaf(root_path, path):
        if (not (depth is None)) and basic.relative_depth(root_path, path) == depth:
            return True
        else:
            candidate_child_paths = get_child_paths(path)
            actual_child_paths = [child_path for child_path in candidate_child_paths if is_child(root_path, child_path)]
            return len(actual_child_paths) == 0

    return crawl(paths, is_leaf, is_child, f, mapper)

def build_build_notebook_log(log_folder, path):

    def build_notebook_log(f):

        import inspect, os
#        path = inspect.getfile(inspect.stack()[1][0])
#        print path, type(path), dir(inspect.getframeinfo(inspect.stack()[1][0]))
#        pdb.set_trace()
        path_folder, path_name = os.path.split(path)
        import python_utils.python_utils.caching as caching
        caching.mkdir(log_folder, False)
#        log_path = '%s/%s-out.ipynb' % (log_folder, path_name)
        log_path = '%s/%s-out.ipynb' % (path_folder, path_name)
        print(log_path)
        import python_utils.python_utils
        import python_utils.python_utils.nbrun as nbrun
        generic_notebook_folder, _ = os.path.split(python_utils.python_utils.__file__)
        generic_notebook_path = '%s/generic_runner.ipynb' % generic_notebook_folder

        def wrapped():
            nbrun.run_notebook(generic_notebook_path, out_path_ipynb=log_path, nb_kwargs={'path':path}, hide_input=True, insert_pos=1, timeout=-1)


        return wrapped

    return build_notebook_log

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        assert False

def intersect_dfs(dfs):
    keys = iter(dfs).next().index
    for df in dfs[1:]:
        keys = keys.intersection(df.index)
    return [df.loc[keys] for df in dfs]

def my_arange(offset, step, total):
    ans = np.arange(offset, abs(step), total)
    if step < 0:
        return reversed(ans)
    else:
        return ans

def get_arg_names(f):
    import inspect
    if isinstance(f, types.FunctionType):
        return inspect.getargspec(f).args
    elif isinstance(f, partial):
        return [arg_name for arg_name in get_arg_names(f.f) if not (arg_name in f.kwargs)]
    else:
        raise NotImplementedError
#        arg_name for arg_name in get_argnames(f.func) if (not (arg_name in f.keywords)) and (not (arg_name in))

class partial(object):

    def __init__(self, f, **kwargs):
        self.f, self.kwargs = f, kwargs
        
    def __call__(self, *args, **kwargs):
        # get argument names of f, fill in based on kwargs, and assume remaining arguments are in order
        arg_names = get_arg_names(self.f)
        _kwargs = {}
        arg_pos = 0
        for arg_name in arg_names:
            if arg_name in self.kwargs:
                _kwargs[arg_name] = self.kwargs[arg_name]
            elif arg_pos < len(args):
                _kwargs[arg_name] = args[arg_pos]
                arg_pos += 1
        for (key,val) in kwargs.iteritems():
            _kwargs[key] = val
        return self.f(**_kwargs)

class my_partial(object):

    def __init__(self, f, supply_arg_names=None, **kwargs):
        import inspect
        self.kwargs, self.f, self.supply_arg_names = kwargs, f, supply_arg_names
#        pdb.set_trace()
        self.arg_names = inspect.getargspec(f).args

    def set_kwarg(self, key, val):
        self.kwargs[key] = val

    def __call__(self, *args):
        if self.supply_arg_names is None:
            arg_names = self.arg_names[-len(args):]
        else:
            arg_names = self.supply_arg_names
        kwargs = {}
        for key in self.kwargs:
            kwargs[key] = self.kwargs[key]
        for (arg,arg_name) in zip(args, arg_names):
            kwargs[arg_name] = arg
        return self.f(**kwargs)
#        try:
#            return self.f(**kwargs)
#        except:
#            pdb.set_trace()
    
def get(obj=None, attr=None, default=None, expr=None):
    if not (expr is None):
        try:
            return expr()
        except AttributeError:
            return default
    try:
        return getattr(obj, attr)
    except AttributeError:
        return default

def add_to_hierarchical_df(key_d, val_d, df):
    # add levels to index
    key_d = {key:val for (key,val) in key_d.items() if (key != 'not_kwargs') and (not (val is None))}
    if df.size > 0:
        for key in key_d.keys():
            if not (key in df.index.names):
                df = pd.concat([df], keys=[''], names=[key])
    else:
        try:
            df = pd.DataFrame(index=pd.MultiIndex(levels=[[] for key in key_d.keys()], labels=[[] for key in key_d.keys()], names=[key for key in key_d.keys()]))
        except:
            df = pd.DataFrame(index=pd.MultiIndex(levels=[[] for key in key_d.keys()], codes=[[] for key in key_d.keys()], names=[key for key in key_d.keys()]))
    tuple_key = tuple([key_d[key] if key in key_d else '' for key in df.index.names])
    for (key, val) in val_d.items():
#        pdb.set_trace()
        if not tuple_key in df.index:
            df.loc[tuple_key,:] = np.nan
        if not key in df.columns:
            df.insert(0, key, [np.nan for _ in range(len(df))])
            if isinstance(val, np.ndarray):
                df = df.astype({key:object})
#        df.at[tuple_key, key] = val
        #df.loc[tuple_key, key] = val
        #print df
        try:
            df[key][tuple_key] = val
        except:
            df = df.astype({key:'O'})
            df[key][tuple_key] = val
        #try:
        #    df.loc[key][tuple_key] = val
#            df.loc[tuple_key, key] = val
        #except:
        #    pdb.set_trace()
        #if (not tuple_key in df.index) and (not key in df.columns):
        #    df.loc[tuple_key, key] = np.nan
        #    df.loc[tuple_key, key] = val
        #else:
        #    pdb.set_trace()
        #    df.loc[tuple_key, key] = val
    return df

def get_from_hierarchical_df(key_d, df):
    tuple_key = tuple([key_d[key] if key in key_d else '' for key in df.index.names])
    return df.loc[tuple_key]


class results(object):

    def __init__(self):
        self.df = pd.DataFrame()

    def add(self, val_d, **kwargs):
        self.df = add_to_hierarchical_df(kwargs, val_d, self.df)

    def get(self, key, **kwargs):
        try:
            ans = get_from_hierarchical_df(kwargs, self.df)[key]
            try:
                if np.isnan(ans):
                    return None
                else:
                    return ans
            except:
                return ans
        except KeyError:
            return None

    def relevant_df(self):
        return self.df.loc[:,self.df.dtypes != object]
