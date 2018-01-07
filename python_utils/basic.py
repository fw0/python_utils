import functools
import json
import types
import decorators
import numpy as np
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm
import time


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
    import StringIO
    from IPython.display import display
    from IPython.display import Image
    output = StringIO.StringIO()
    fig.savefig(output, format='png')
    img = Image(output.getvalue())
    display(img)
    import matplotlib.pyplot as plt
#    import caching
#    caching.fig_archiver.archive_fig(fig)
    plt.close()
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
        shutil.rmtree(self.folder)


    def __call__(self, fig):
        import os, pdb
        try:
            os.makedirs(self.folder)
        except Exception,e:
            print e
        fig.savefig('%s/%d' % (self.folder, self.counter))
        self.counter += 1
        return fig


class archiver(object):
    
    def __init__(self, folder):
        import os, pdb
        self.folder = folder
        self.counter = 0
        try:
            os.makedirs(self.folder)
        except Exception,e:
            print e
        self.f = open('%s/log.txt' % self.folder, 'w')

    def archive_fig(self, fig):
        if True:
            display_fig_inline(fig)
            fig.savefig('%s/%d' % (self.folder, self.counter))
            self.counter += 1
            return fig

    def archive_string(self, s):
        f = open('%s/%d.txt' % (self.folder, self.counter), 'w')
        f.write(s)
        self.counter += 1
        print s
        f.close()
        return s

    def log_text(self, *s):
        if True:
            print s
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
        

import copy_reg
import types
import multiprocessing


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def joblib_parallel_map(num_processes, verbose, f, iterable):

    import joblib
    return joblib.Parallel(n_jobs=num_processes, verbose=verbose)(joblib.delayed(f)(x) for x in iterable)
#    return joblib.Parallel(n_jobs=num_processes, verbose=verbose, backend='threading')(joblib.delayed(f)(x) for x in iterable)

        
def parallel_map(num_processes, f, iterable):
    """                                                                                                                                                                                                    
    make a                                                                                                                                                                                                 
    """
    import multiprocessing
    #print iterable
    print f
    results = multiprocessing.Manager().list()
    iterable_queue = multiprocessing.Queue()

    def worker(_iterable_queue, _f, results_queue):
        for arg in iter(_iterable_queue.get, None):
            results_queue.append(_f(arg))

    for x in iterable:
        iterable_queue.put(x)

    for i in xrange(num_processes):
        iterable_queue.put(None)

    workers = []

    for i in xrange(num_processes):
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

#            print '%r %4.4f sec' % \
#                (method.__name__, te-ts), msg, 'gg'
            import pdb
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

import cProfile, pstats, StringIO

def do_cprofile(sort_by):
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                s = StringIO.StringIO()
                print sort_by
                import pdb
#                ps = pstats.Stats(profile, stream=s).strip_dirs().sort_stats(sort_by)
                ps = pstats.Stats(profile, stream=s).sort_stats(sort_by)
                ps.print_stats()
                print s.getvalue()
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
    print target, source
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

def basic_map_getter(mode, profile='', splat=False):
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

    def splat_mapper(f, iterable):
        def dec_f(args):
            return f(*args)
        return mapper(dec_f, iterable)

    return mapper if splat else splat_mapper
    
def map_getter(mode, profile, compute, recompute):

    def _constructor(reader=None, writer=None, get_path=None, suffix=None):
        
        
        import python_utils.python_utils.caching as caching
        mapper = basic_map_getter(mode, profile)
        mapper = caching.igmore_exception_map_wrapper(mapper)
    
        def final(f, iterable):
            decorated = caching.switched_decorator(f, compute, recompute, reader, writer, get_path, suffix)
            return mapper(decorated, iterable)
    
        return final

    return _constructor

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
