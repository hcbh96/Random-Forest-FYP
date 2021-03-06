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
exec(compile(open('drop_var_auc_roc.py', "rb").read(), 'drop_var_auc_roc.py', 'exec'))
exec(compile(open('most_important_vars.py', "rb").read(), 'most_important_vars.py', 'exec'))
exec(compile(open('number_of_features.py', "rb").read(), 'number_of_features.py', 'exec'))
exec(compile(open('permutations_eda.py', "rb").read(), 'permutations_eda.py', 'exec'))
exec(compile(open('casa_vs_other.py', "rb").read(), 'casa_vs_other.py', 'exec'))


