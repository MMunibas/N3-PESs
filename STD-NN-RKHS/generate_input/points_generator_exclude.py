import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.optimize import curve_fit
import random

path = '/home/wjchun/work_2023/O2-project/PESJuan/Data2/'
#path2 = '../../../Data/'
plotting = False
external_plotting = False

Egridout = []
i = 0.0
while (i+0.2) < 1.0:
    Egridout.append(i)
    i = i + 0.2
while (i+0.4) < 6.0:
    Egridout.append(i)
    i = i + 0.4
while (i+0.2) < 9.8:
    Egridout.append(i)
    i = i + 0.2
while (i+0.3) < 16.0:
    Egridout.append(i)
    i = i + 0.3
Egridout.append(16.0)
Egridout = np.array(Egridout)


vgridout = []
i = 0
while (i + 1) < 57:
    vgridout.append(i)
    i = i + 1
vgridout.append(57)
vgridout = np.array(vgridout)

jgridout = []
i = 0
while (i + 6) < 272:
    jgridout.append(i)
    i = i + 6
jgridout.append(272)
jgridout = np.array(jgridout)


num_features = 11
num_outputs = len(Egridout) + len(vgridout) + len(jgridout)
print('Number of features: ' + str(num_features))
print('Number of outputs: ' + str(num_outputs))

filenames = ['pevj', 'pv', 'pj']
features = np.genfromtxt('input_new.txt', delimiter=',')


with open("indices_normal_prob.txt", "w") as txt_file:
    for i in range(len(features)):
        E, p = np.loadtxt(path + 'pv' + str(i+1) + '.dat', unpack=True)
        if np.sum(p) > 0.02:
            txt_file.write(str(i) + '\n')

with open("indices_low_prob.txt", "w") as txt_file:
    for i in range(len(features)):
        E, p = np.loadtxt(path + 'pv' + str(i+1) + '.dat', unpack=True)
        if not (np.sum(p) > 0.02):
            txt_file.write(str(i) + '\n')



