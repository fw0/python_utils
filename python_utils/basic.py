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

    def __call__(self, fig):
        import os, pdb
        try:
            os.makedirs(self.folder)
        except Exception,e:
            print e
        fig.savefig('%s/%d' % (self.folder, self.counter))
        self.counter += 1
        return fig

def compose(f, g):

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h
        
    
def get_grid_fig_axes(n_rows, n_cols, n):
    import matplotlib.pyplot as plt
    per_fig = n_rows * n_cols
    num_pages = (int(n) / per_fig) + (1 if (n % per_fig) > 0 else 0)
    figs = []
    axes = []
    for i in xrange(num_pages):
        start = per_fig * i
        end = min(per_fig * (i+1), n)
        assert (end - start) > 0
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


def set_legend_unique_labels(ax, *args, **kwargs):
    import operator
    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = zip(*(dict(zip(labels, handles)).iteritems()))
    hl = sorted(zip(unique_handles, unique_labels), key=operator.itemgetter(1))
    sorted_handles, sorted_labels = zip(*hl)
    ax.legend(sorted_handles, sorted_labels, **kwargs)


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

def vals_to_rgbas(vals):

    def get_elt_rank(l):
        ans = np.zeros(len(l))
        for (i,pos) in enumerate(np.argsort(l)):
            ans[pos] = i
        return np.array(ans)

    vals = get_elt_rank(vals) / len(vals)
    c = mpl_colors.Normalize()
    c.autoscale(vals)
    m = cm.ScalarMappable(norm=c, cmap=cm.cool)
    colors = map(tuple,m.to_rgba(vals))
    return colors
