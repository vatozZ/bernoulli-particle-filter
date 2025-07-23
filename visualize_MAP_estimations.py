import json
import matplotlib.pyplot as plt
import numpy as np


with open('bernoulli_pf_parametre_1/bernoulli_pf_parametre_1_simulation_records.json', 'r') as f:
    data1 = json.load(f)

ospa_parametre_1 = data1['ospa_MAP_records']

with open('bernoulli_pf_parametre_2/bernoulli_pf_parametre_2_simulation_records.json', 'r') as f:
    data2 = json.load(f)

ospa_parametre_2 = data2['ospa_MAP_records']

n_particles1 = data1['n_particles']
n_particles2 = data2['n_particles']

plt.plot(ospa_parametre_1, label=str(n_particles1))
plt.xlabel('Discrete time - k')
plt.savefig('ospa_bernoulli_pf.png', dpi=1100, bbox_inches='tight')
plt.show()

"""plt.plot(data['qk_k'])
plt.xlabel('Discrete time -k')
plt.ylabel(r'$q_k$')
#plt.savefig('qk_existence.png', dpi=1100, bbox_inches='tight')
plt.show()"""