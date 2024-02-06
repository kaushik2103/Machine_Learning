import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

# Using SweetViz Library
import sweetviz as sv
my_report = sv.analyze(df)
my_report.show_html()

# Using dtale Library
import dtale
import dtale.app as dtale_app

dtale_app.USE_COLAB = True
dtale.show(df)