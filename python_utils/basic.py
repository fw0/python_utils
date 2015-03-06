import functools
import json
import types

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

def is_iterable(obj):
    try:
        temp = iter(obj)
    except TypeError:
        return False
    else:
        return True

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
        fig.tight_layout()
        figs.append(fig)
    return figs, axes
