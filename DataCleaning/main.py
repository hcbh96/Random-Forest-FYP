"""
This file can be used to run and produce all files
graph associated with Optimising Livestock Production
"""

exec(compile(open('clean.py', "rb").read(), 'clean.py', 'exec'))
exec(compile(open('pairplots.py', "rb").read(), 'pairplots.py', 'exec'))
exec(compile(open('correlation_heatmap.py', "rb").read(), 'correlation_heatmap.py', 'exec'))
exec(compile(open('decision_tree.py', "rb").read(), 'decision_tree.py', 'exec'))
exec(compile(open('unoptimised_forest.py', "rb").read(), 'unoptimised_forest.py', 'exec'))
exec(compile(open('optimised_forest.py', "rb").read(), 'optimised_forest.py', 'exec'))
exec(compile(open('roc_auc.py', "rb").read(), 'roc_auc.py', 'exec'))
