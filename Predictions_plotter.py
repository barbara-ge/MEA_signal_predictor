import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import read_csv
import plotly.graph_objs as go
import plotly.offline
import plotly.express as px

directory = '/Volumes/Elements/spring2022/final/week4/Predictor/'
pred_NS = 'Pred_prediction_NS.csv'
pred_8020 = 'Pred_prediction_8020.csv'
pred_5050 = 'Pred_prediction_5050.csv'
real_NS = 'Real_prediction_NS.csv'
real_8020 = 'Real_prediction_8020.csv'
real_5050 = 'Real_prediction_5050.csv'
r2_NS = 'R2_prediction_NS.csv'
r2_8020 = 'R2_prediction_8020.csv'
r2_5050 = 'R2_prediction_5050.csv'


p_NS = read_csv(directory + pred_NS)
p_NS = p_NS.values

p_8020 = read_csv(directory + pred_8020)
p_8020 = p_8020.values

p_5050 = read_csv(directory + pred_5050)
p_5050 = p_5050.values

r_NS = read_csv(directory + real_NS)
r_NS = r_NS.values

r_8020 = read_csv(directory + real_8020)
r_8020 = r_8020.values

r_5050 = read_csv(directory + real_5050)
r_5050 = r_5050.values

r2_NS_d = read_csv(directory + r2_NS)
r2_NS_d = r2_NS_d.values

r2_8020_d = read_csv(directory + r2_8020)
r2_8020_d = r2_8020_d.values

r2_5050_d = read_csv(directory + r2_5050)
r2_5050_d = r2_5050_d.values

#plot 1
tot_r2 = [r2_NS_d, r2_8020_d, r2_5050_d]
#tot_r2 = np.asarray(tot_r2)

fig, ax = plt.subplots(figsize=(10,7))
ax.boxplot([r2_NS_d.flatten(), r2_8020_d.flatten(), r2_5050_d.flatten()])
ax.set_xticklabels(['NS', '8020', '5050'])
ax.set_ylabel('R$^{2}$')
plt.ylim([0, 1])
fig.show()
fig.savefig(directory + 'R2.pdf', dpi=300)
plt.close()

# fig = go.Figure()
# fig.add_trace(go.Box(y=r2_NS_d, quartilemethod="linear", name="NS"))
# fig.add_trace(go.Box(y=r2_8020_d, quartilemethod="linear", name="8020"))
# fig.add_trace(go.Box(y=r2_5050_d, quartilemethod="linear", name="5050"))
#
# fig.show()
m, b = np.polyfit(r_NS.flatten(), p_NS.flatten(), 1)
m2, b2 = np.polyfit(r_8020.flatten(), p_8020.flatten(), 1)
m3, b3 = np.polyfit(r_5050.flatten(), p_5050.flatten(), 1)

fig, ax = plt.subplots(figsize=(10,7))
#ax.scatter(r_NS, p_NS, label='NS - m = {:.2f}'.format(m),  alpha=0.3, edgecolors='none')
ax.plot(r_NS, p_NS, 'o', color = 'blue', label='NS - m = {:.2f}'.format(m))
ax.plot(r_NS, m*r_NS+b, color = 'blue')
#ax.scatter(r_8020, p_8020, label='8020 - m = {:.2f}'.format(m2), alpha=0.3, edgecolors='none')
ax.plot(r_8020, p_8020, 'o', color = 'magenta', label='8020 - m = {:.2f}'.format(m2))
ax.plot(r_8020, m2*r_8020+b2, color = 'magenta')
#ax.scatter(r_5050, p_5050, label='5050 - m = {:.2f}'.format(m3), alpha=0.3, edgecolors='none')
ax.plot(r_5050, p_5050, 'o', color = 'yellow', label='5050 - m = {:.2f}'.format(m3))
ax.plot(r_5050, m3*r_5050+b3, color = 'yellow')
ax.legend()
ax.set_ylabel('Predicted')
ax.set_xlabel('Real')
plt.show()
fig.savefig(directory + 'predVSreal.pdf', dpi=300)
plt.close()