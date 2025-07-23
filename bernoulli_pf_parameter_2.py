from numpy.random import uniform, poisson, normal
import numpy as np
from numpy import array, random
from copy import deepcopy
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import json
from ospa import OSPAMetric

ospa_MMSE_records = []
ospa_MAP_records = []
ospa_Lk_based_records = []
OSPA = OSPAMetric()

simulation_records = {}

n_simulation = 20

N_particles = 5
n_state_dims = 4

N_birth = 1

Pb = 0.2
Ps = 0.8
Pd = 0.9

qk_km1_init = 0.01

x0, xf = 0, 30
y0, yf = 0, 30

Vmin, Vmax = 0, 1

sensor_noise_std = 0.1
process_noise_std = 0.1

dt = 1.0
poisson_lambda = 5  # clutter mean for the clutter model
lambda_clutter = poisson_lambda / ((xf - x0) * (yf - y0))

x_init, y_init, Vx_init, Vy_init = 1, 1, 1, 1
track_init, track_terminate = 5, 15  # track starts appearing, disappears
gt = np.array([x_init, y_init, Vx_init, Vy_init]).reshape(-1, 1)
Fk = np.array([[1, 0, dt, 0],
               [0, 1, 0, dt],
               [0, 0, 1, 0],
               [0, 0, 0, 1]]).reshape(4, 4)

simulation_records['Ps'] = Ps
simulation_records['Pb'] = Pb
simulation_records['Pd'] = Pd
simulation_records['n_particles'] = N_particles
simulation_records['n_birth'] = N_birth
simulation_records['sensor_noise_std'] = sensor_noise_std
simulation_records['process_noise_std'] = process_noise_std
simulation_records['poisson_lambda'] = poisson_lambda
simulation_records['lambda_clutter'] = lambda_clutter
simulation_records['track_init'] = track_init
simulation_records['track_terminate'] = track_terminate
simulation_records['qk_km1_init'] = qk_km1_init
simulation_records['n_simulation'] = n_simulation

Zk = []
GT = []
for sim_i in range(n_simulation):
    zk = []
    mkc = poisson(poisson_lambda)
    for ck in range(mkc):
        ck_x = uniform(x0, xf)
        ck_y = uniform(y0, yf)
        zk.append([ck_x, ck_y])

    # creating the observation list
    if track_init < sim_i < track_terminate:

        gt = Fk @ gt  # np matmul  il çarp
        sensor_xk = deepcopy(gt.flatten().tolist()[:2])

        sensor_xk[0] = sensor_xk[0] + normal(0, sensor_noise_std)
        sensor_xk[1] = sensor_xk[1] + normal(0, sensor_noise_std)

        zk.append(sensor_xk)
        GT.append(sensor_xk)
    else:
        GT.append([])

    random.shuffle(zk)

    Zk.append(zk)

simulation_records['Zk'] = Zk
simulation_records['GT'] = GT

def reset():
    # Uzamsal Dağılım

    Xkk = array([uniform(low=x0, high=xf, size=N_particles),
                 uniform(low=y0, high=yf, size=N_particles),
                 uniform(low=Vmin, high=Vmax, size=N_particles),
                 uniform(low=Vmin, high=Vmax, size=N_particles)]).reshape(n_state_dims, N_particles)

    wkk = np.ones(N_particles) / N_particles

    # Doğum Bölgelerinin Dağılımı
    Xkk = np.hstack((Xkk, array([5, 5, 1, 1]).reshape(n_state_dims, 1)))
    wkk = np.hstack((wkk, array([1.])))

    plt.figure()
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    plt.scatter(Xkk[0, :], Xkk[1, :], c='darkviolet')
    plt.scatter(Xkk[0, 1], Xkk[1, 1], c='darkviolet', label='particles')
    plt.legend(framealpha=1)
    plt.savefig('bernoulli_pf_parametre_2/initialization.png', dpi=1100, bbox_inches='tight')
    plt.clf()

    return wkk, Xkk


def predict(qkm1_km1, Xkk, wkk, dt, N_adaptive_birth):
    # Var olma olasılığı hesapla
    qk_km1 = Pb * (1 - qkm1_km1) + Ps * qkm1_km1

    # Uzamsal dağılımın ağırlıklarını tahmin et

    for i in range(Xkk.shape[1] - N_adaptive_birth):
        wkk[i] = wkk[i] * (Ps * qkm1_km1) / qk_km1

    for j in range(-N_adaptive_birth):
        wkk[N_particles + j] = Pb * (1 / N_adaptive_birth) * (1 - qkm1_km1) / qk_km1

    Fkm1 = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]).reshape(4, 4)

    Xkk = Fkm1 @ Xkk
    wkk = deepcopy(wkk)

    """
    Normalize the weights...
    """

    wk_sum = 0.0
    for wk_i in wkk:
        wk_sum = wk_sum + wk_i

    for index_k, wk_kk in enumerate(wkk):
        wkk[index_k] = wk_kk / wk_sum

    return Xkk, wkk, qk_km1


def update(Zk, Xk_km1, wk_km1, qk_km1):

    I1 = 0.0  # equation 84
    for wk_j in wk_km1:
        I1 = I1 + Pd * wk_j

    Hk = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]]).reshape(2, 4)

    I2 = 0.0  # equation 85
    for zki in Zk:
        for index_i, particle_i in enumerate(Xk_km1.T):

            # equation 85
            xk_state = np.matmul(Hk, particle_i)
            R = np.diag([sensor_noise_std, sensor_noise_std]) #(0.1, 0.1)
            lk = mvn(xk_state, R).pdf(zki)
            I2 = I2 + Pd * lk * wk_km1[index_i]

    delta_k = I1 - I2 / lambda_clutter

    # Probability of Existence
    qk_k = (1 - delta_k) / (1 - delta_k * qk_km1) * qk_km1

    # Ağırlıkları Güncelle (denklem 87
    lk_array_ = []
    for j, particle_j in enumerate(Xk_km1.T):
        lk_total = 0.0
        for zk_j in Zk:
            xk_state = Hk @ particle_j
            R = np.diag([sensor_noise_std, sensor_noise_std])
            lk = mvn(xk_state, R).pdf(zk_j)
            lk_total = lk_total + lk
        lk_array_.append(lk_total)
        wk_km1[j] = (1 - Pd + Pd * lk_total / lambda_clutter) * wk_km1[j]

    wk_k_total = 0.0
    for wk_i in wk_km1:
        wk_k_total = wk_k_total + wk_i

    for index_i, wk_ii in enumerate(wk_km1):
        wk_km1[index_i] = wk_ii / wk_k_total

    # RESAMPLING ##

    if np.sum(wk_km1) > 0.0:

        idx_max = np.argmax(wk_km1)
        Xkk_max = Xk_km1[:, idx_max]
        n_particle_ = Xkk_init.shape[1]  # 201
        #wk_km1[wk_km1 < 0] = 0.0
        idx = random.choice(Xk_km1.shape[1], Xk_km1.shape[1], replace=True, p=wk_km1)
        Q = [process_noise_std, process_noise_std, 0.1, 0.1]
        Q = np.diag(Q)
        n_state_dims = Xk_km1.shape[0]
        resample_rv = mvn(mean=np.zeros(n_state_dims), cov=Q)
        Xkk = Xk_km1[:, idx] + resample_rv.rvs(size=Xk_km1.shape[1]).T
        Xkk[2, :] = np.abs(Xkk[2, :])
        Xkk[2, :] = np.where(Xkk[2, :] < 0.1, Xkk[2, :], 0.1)  # map Velocity range to: (min_Velocity, max_Velocity)
        #Xkk[:, 0] = Xkk_max

        """n_brut_force = 1
        for n_i in range(n_brut_force):
            Xkk[:, n_i] = Xkk_max"""

        wkk = wk_km1
        wkk[:] = 1 / n_particle_

    else:
        wkk, Xkk = reset()
        lk_array_ = wkk

    """
    Adaptive Birth
    """

    N_adaptive_birth = len(Zk)

    for z in Zk:
        Xkk = np.hstack((Xkk, array([z[0], z[1], 1., 1.]).reshape(4, 1)))
        wkk = np.hstack((wkk, array([1. / N_adaptive_birth])))

    return wkk, Xkk, qk_k, lk_array_, N_adaptive_birth


def estimate_target_state(Xkk, wkk, lk, N_adaptive_birth):
    max_idx = np.argmax(lk)
    Xkk_no_birth = deepcopy(Xkk)

    Xkk_no_birth = Xkk_no_birth[:, :-N_adaptive_birth]

    Xkk_MAP = Xkk_no_birth[:, max_idx]

    Xkk_MMSE = np.matmul(Xkk, wkk)

    return Xkk_MAP, Xkk_MMSE


wkk_init, Xkk_init = reset()

wk_k = wkk_init
Xk_k = Xkk_init
qk_k = qk_km1_init
qk_km1 = qk_km1_init

qk_records = []
qkm1_records = []

map_records = []
mmse_records = []

N_adaptive_birth = 1

ospa_MMSE_records = []
ospa_MAP_records = []

for i in range(n_simulation):

    print("** simulation: {}".format(i))

    Xk_km1, wk_km1, qk_km1 = predict(qkm1_km1=qk_k, Xkk=Xk_k, wkk=wk_k, dt=1, N_adaptive_birth=N_adaptive_birth)

    wk_k, Xk_k, qk_k, lk_array, N_adaptive_birth = update(Zk=Zk[i], Xk_km1=Xk_km1, wk_km1=wk_km1, qk_km1=qk_km1)

    MAP, MMSE = estimate_target_state(Xkk=Xk_k, wkk=wk_k, lk=lk_array, N_adaptive_birth=N_adaptive_birth)

    print("qk_k", round(qk_k, 1), round(qk_km1, 1))
    """
    Plot figure
    """

    qk_records.append(qk_k)
    qkm1_records.append(qk_km1)

    mmse_records.append(list(MMSE.flatten()))
    map_records.append(list(MAP.flatten()))

    ospa_dist_MMSE = OSPA.compute_OSPA_distance(track_states=list(MMSE[:2].flatten()), truth_states=GT[i])

    ospa_dist_MAP = OSPA.compute_OSPA_distance(track_states=list(MAP[:2].flatten()), truth_states=GT[i])
    print(ospa_dist_MAP, ospa_dist_MMSE)

    ospa_MMSE_records.append(ospa_dist_MMSE)
    ospa_MAP_records.append(ospa_dist_MAP)

    plt.clf()
    plt.xlim([x0, xf])
    plt.ylim([y0, yf])

    plt.scatter(Xk_k[0, :], Xk_k[1, :], s=10, c='darkviolet')
    plt.scatter(Xk_k[0, 0], Xk_k[1, 0], s=10, c='darkviolet', label='particles')

    if len(GT[i]) != 0:
        plt.scatter(GT[i][0], GT[i][1], c='green', marker='s', s=90, label='GT')

    for zk_i in Zk[i]:
        plt.scatter(zk_i[0], zk_i[1], c='red')
    plt.scatter(Zk[i][-1][0], Zk[i][-1][1], c='red', label='detections')

    plt.scatter(MAP[0], MAP[1], label='estimate', c='black', s=70, marker='s')

    plt.xlabel('x[m]')
    plt.ylabel('y[m]')
    #plt.title("simulation:" + str(i) + "  number of particle:" + str(Xk_k.shape[1]))
    plt.legend()
    plt.savefig('bernoulli_pf_parametre_2/bernoulli_particle_filter_spatial_' + str(i) + '.png', dpi=1100, bbox_inches='tight')
    plt.pause(0.01)

simulation_records['qk_k'] = qk_records
simulation_records['qkm1'] = qkm1_records
simulation_records['map'] = map_records
simulation_records['mmse'] = mmse_records
simulation_records['ospa_MAP_records'] = ospa_MAP_records

plt.figure()
plt.plot(ospa_MAP_records, linestyle=':')
plt.xlabel('Discrete time - k')
plt.savefig('bernoulli_pf_parametre_2/ospa_results.png', dpi=1100, bbox_inches='tight')
plt.show()

plt.clf()
for z in Zk:
    for zj in z:
        plt.scatter(zj[0], zj[1], c='red')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.savefig('bernoulli_pf_parametre_2/detections.png',dpi=1100, bbox_inches='tight')
plt.show()

plt.clf()
plt.plot(qk_records)
plt.ylabel(r'$q_{k|k}$')
plt.xlabel('Discrete time - k')
plt.savefig('bernoulli_pf_parametre_2/bernoulli_particle_filter_qkk.png', dpi=1100, bbox_inches='tight')
plt.show()

with open('bernoulli_pf_parametre_2/bernoulli_pf_parametre_2_simulation_records.json', 'w') as json_file:
    json.dump(simulation_records, json_file)