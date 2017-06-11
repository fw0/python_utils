import functools
import json
import types
import decorators
import numpy as np
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm

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
        print s
        self.f.write(str(s)+'\n')
        self.f.flush()

import copy_reg
import types
import multiprocessing


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)
        
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

    def __call__(self, f):
        
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            raise Exception

        return wrapped_f
    
class raise_exception_method(decorators.decorated_method):

    def __call__(self, inst, *args, **kwargs):
        raise Exception

    
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

import cProfile

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

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
