import seaborn as sns

df = sns.load_dataset('titanic')
import sweetviz as sv

my_report = sv.analyze(df)
my_report.show_html()
