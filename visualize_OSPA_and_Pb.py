import json
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import matplotlib.ticker as ticker

data = json.load(open('bpf_monte_carlo_simulation.json'))

ospa_keys = []

ospa_records = []

for key in data.keys():
    if key.split('_')[0] == 'ospa':
        ospa_records.append(data[key])

full_mean_ospa = np.zeros(shape=(20, 1))

for one_scan in ospa_records:

    for each, item in enumerate(one_scan):
        full_mean_ospa[each] = full_mean_ospa[each] + item

full_mean_ospa = (full_mean_ospa / 100).reshape(1, 20)

full_mean_ospa = full_mean_ospa.tolist()[0]

plt.plot(full_mean_ospa)
plt.xlabel('Discrete time - k')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#plt.savefig('monte_carlo_100_run_ospa.png', dpi=1100, bbox_inches='tight')
plt.clf()
#plt.show()
qk_keys = []

qk_records = []

for key in data.keys():
    if key.split('_')[0] == 'qk':
        if key.split('_')[1] == 'km1':
            continue
        qk_records.append(data[key])

full_mean_qk = np.zeros(shape=(20, 1))

for one_scan in qk_records:
    for each, item in enumerate(one_scan):
        full_mean_qk[each] = full_mean_qk[each] + item

full_mean_qk = (full_mean_qk / 100).reshape(1, 20)

full_mean_qk = full_mean_qk.tolist()[0]


x = [0.5] * 20
x[0] = x[1] = x[2] = x[18] = x[19] = 0

plt.plot(x, label='GT')
plt.plot(full_mean_qk, label=r'$q_k$')
plt.xlabel('Discrete time - k')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylim([0, 1])
plt.savefig('monte_carlo_100_run_qk.png', dpi=1100, bbox_inches='tight')

plt.show()




