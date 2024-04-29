import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.optimize import curve_fit
import random

path = '../DataAE/'
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
while (i + 1) < 47:
    vgridout.append(i)
    i = i + 1
vgridout.append(47)
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
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
pdfE = matplotlib.backends.backend_pdf.PdfPages("output_E.pdf")
pdfv = matplotlib.backends.backend_pdf.PdfPages("output_v.pdf")
pdfj = matplotlib.backends.backend_pdf.PdfPages("output_j.pdf")
# pdfall = matplotlib.backends.backend_pdf.PdfPages("output_all.pdf")
features = np.genfromtxt('input_new.txt', delimiter=',')
normal_data = np.genfromtxt('indices_normal_prob.txt', delimiter=',').astype(int)
low_prob_data = np.genfromtxt('indices_low_prob_not_excluded.txt', delimiter=',').astype(int)
excluded_data = np.genfromtxt('indices_to_exclude.txt', delimiter=',').astype(int)


# seed = 33
# random.seed(seed)
# indices = list(total_data)
# test_indices = []
# with open("test_indices.txt", "w") as txt_file:
#     for i in range(56):
#         index = random.choice(indices)
#         indices.remove(index)
#         test_indices.append(index)
#         txt_file.write(str(index) + '\n')


# with open("indices_normal_prob.txt", "w") as txt_file:
#     for i in range(2184):
#         E, p = np.loadtxt(path + 'pv' + str(i+1) + '.dat', unpack=True)
#         if np.sum(p) > 0.005:
#             txt_file.write(str(i) + '\n')
#
# with open("indices_low_prob.txt", "w") as txt_file:
#     for i in range(2184):
#         E, p = np.loadtxt(path + 'pv' + str(i+1) + '.dat', unpack=True)
#         if not (np.sum(p) > 0.005):
#             txt_file.write(str(i) + '\n')


# quit()
# eV_to_joule = 1.602176565*(10**(-19))
# hbar = 1.054571817*(10**(-34))
# N_avo = 6.02214086*(10**(23))
# m_O2 = 2*15.9994
# m_N = 14.0067
# mu_rel = (m_O2*m_N)/(m_O2+m_N)
# features = np.genfromtxt('O2_states.dat', skip_header=1)
#
# v = features[:, 0]
# j = features[:, 1]
# Evj = features[:, 3]
# turn_l = features[:, 5]
# turn_r = features[:, 4]
# period = features[:, 6]
#
# data = np.genfromtxt(path + 'nevjinput.dat', delimiter=' ')
#
#
# with open("input_new.txt", "w") as txt_file_main:
#     for i in range(2184):
#         for k in range(len(Evj)):
#             if v[k] == int(data[i, 0]) and j[k] == int(data[i, 1]):
#                 Etrans_curr = data[i, 2]
#                 vel_curr = np.sqrt(2*Etrans_curr/mu_rel)
#                 v_curr = v[k]
#                 j_curr = j[k]
#                 angular_mom_curr = np.sqrt(j_curr*(j_curr+1))
#                 Evj_curr = Evj[k]
#                 turn_l_curr = turn_l[k]
#                 turn_r_curr = turn_r[k]
#                 period_curr = period[k]
#             if v[k] == int(data[i, 0]) and j[k] == 0:
#                 Ev_curr = Evj[k]
#             if v[k] == 0 and j[k] == int(data[i, 1]):
#                 Ej_curr = Evj[k]
#
#         txt_file_main.write(str(Etrans_curr) + ',' + str(v_curr) + ',' + str(j_curr) + ',' + str(vel_curr) + ',' + str(Evj_curr) + ',' + str(Ev_curr) +
#                             ',' + str(Ej_curr) + ',' + str(angular_mom_curr) + ',' + str(turn_l_curr) + ',' + str(turn_r_curr) + ',' + str(period_curr) + '\n')
# quit()


def line(x, a, b):
    return a*x+b


def unavg_max(p, x, num_nb, type, low_prob):
    if type == 'pevj':
        if not low_prob:
            a_crit = 0.001
        else:
            a_crit = 0.00055
    elif type == 'pv':
        if not low_prob:
            a_crit = 0.000143
        else:
            a_crit = 0.000017
    elif type == 'pj':
        if not low_prob:
            a_crit = 0.000005
        else:
            a_crit = 1.4e-6

    init_par = [(p[num_nb-1] - p[0])/(x[num_nb-1]-x[0]), ((p[0] + p[num_nb-1]) -
                                                          ((p[num_nb-1] - p[0])/(x[num_nb-1]-x[0]))*(x[0]+x[num_nb-1]))/2]
    fit_par_l, fit_cov_l = curve_fit(line, x[:num_nb], p[:num_nb], init_par)

    init_par = [(p[2*num_nb] - p[num_nb+1])/(x[num_nb+1]-x[2*num_nb]), ((p[2*num_nb] + p[2*num_nb]) -
                                                                        ((p[2*num_nb] - p[num_nb+1])/(x[2*num_nb]-x[num_nb+1]))*(x[num_nb+1]+x[2*num_nb]))/2]
    fit_par_r, fit_cov_r = curve_fit(line, x[num_nb+1:2*num_nb+1], p[num_nb+1:2*num_nb+1], init_par)

    if fit_par_l[0] > a_crit and fit_par_r[0] < -a_crit:
        return True, fit_par_l[0], fit_par_l[1], fit_par_r[0], fit_par_r[1]
    else:
        return False, fit_par_l[0], fit_par_l[1], fit_par_r[0], fit_par_r[1]


with open("input_for_neural_new.txt", "w") as txt_file:
    with open("indices_for_neural.txt", "w") as txt_file_2:
        for i in range(len(features)): #len(features)
            # for i in total_data:
            print('Current set: ' + str(i))

            if i not in excluded_data:
                txt_file_2.write(str(i)+'\n')

                for j in range(11):
                    txt_file.write(str(features[i, j]) + ',')

                if i in low_prob_data:
                    low_prob = True
                else:
                    low_prob = False

                for filename in filenames:
                    E, p = np.loadtxt(path + filename + str(i+1) + '.dat', unpack=True)

                    if filename == 'pevj':

                        grid = Egridout

                        p_full_avg = []
                        for j in range(len(p)):
                            if j-1 < 0 or j+1 > (len(p)-1):
                                p_full_avg.append(p[j])
                            elif j-2 < 0 or j+2 > (len(p)-2):
                                p_full_avg.append(np.mean(p[j-1:j+1+1]))
                            elif j-3 < 0 or j+3 > (len(p)-3):
                                p_full_avg.append(np.mean(p[j-2:j+2+1]))
                            else:
                                p_full_avg.append(np.mean(p[j-3:j+3+1]))

                        p_full_avg_2 = []
                        for j in range(len(p)):
                            if j-1 < 0 or j+1 > (len(p)-1):
                                p_full_avg_2.append(p[j])
                            elif j-2 < 0 or j+2 > (len(p)-2):
                                p_full_avg_2.append(np.mean(p[j-1:j+1+1]))
                            else:
                                p_full_avg_2.append(np.mean(p[j-2:j+2+1]))

                        p_on_grid = []
                        p_on_grid_avg = []
                        p_on_grid_avg_2 = []
                        for ele in grid:
                            p_on_grid.append(p[list(E).index(round(ele, 1))])
                            p_on_grid_avg.append(p_full_avg[list(E).index(round(ele, 1))])
                            p_on_grid_avg_2.append(p_full_avg_2[list(E).index(round(ele, 1))])

                        index_of_max = np.argmax(np.array(p_on_grid))
                        num_nb = 4
                        if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                            unavg_max_bool, a_l, b_l, a_r, b_r = unavg_max(
                                p_on_grid_avg[index_of_max-num_nb:index_of_max+num_nb+1], grid[index_of_max-num_nb:index_of_max+num_nb+1], num_nb, filename, low_prob)
                        else:
                            num_nb = index_of_max
                            if num_nb > 2:
                                unavg_max_bool, a_l, b_l, a_r, b_r = unavg_max(
                                    p_on_grid[index_of_max-num_nb:index_of_max+num_nb+1], grid[index_of_max-num_nb:index_of_max+num_nb+1], num_nb, filename, low_prob)
                            else:
                                unavg_max_bool = True
                                a_l = 0
                                b_l = 0
                                a_r = 0
                                b_r = 0

                        p_on_grid_final_1 = []
                        for j in range(len(grid)):
                            if (index_of_max-2 <= j <= index_of_max+2) and unavg_max_bool:
                                if j == index_of_max:
                                    p_on_grid_final_1.append(p_on_grid[j])
                                else:
                                    p_on_grid_final_1.append(p_on_grid_avg_2[j])
                            else:
                                p_on_grid_final_1.append(p_on_grid_avg[j])

                        p_on_grid_final = []
                        prev = False
                        for j in range(len(p_on_grid_final_1)):
                            if j == 0 or j == len(p_on_grid_final_1)-1 or (j == index_of_max and unavg_max_bool):
                                p_on_grid_final.append(p_on_grid_final_1[j])
                            else:
                                if not prev and p_on_grid_final_1[j] != 0:
                                    p_on_grid_final.append(
                                        (p_on_grid_final_1[j-1] + p_on_grid_final_1[j+1])/2)
                                    prev = True
                                else:
                                    p_on_grid_final.append(p_on_grid_final_1[j])
                                    prev = False

                        for ele in p_on_grid_final:
                            txt_file.write(str(ele) + ',')

                        if plotting:
                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                xrange = np.linspace(grid[index_of_max-2]-0.1,
                                                     grid[index_of_max+2]+0.1, 50)

                            plt.figure()
                            plt.plot(E, p, '-b', label='QCT')
                            plt.plot(grid, p_on_grid_final, '.r', label='Grid')
                            plt.plot(grid[index_of_max], p_on_grid[index_of_max], '.g')
                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                plt.plot(xrange, line(xrange, a_l, b_l), '-k', label='Line1')
                                plt.plot(xrange, line(xrange, a_r, b_r), '-k', label='Line2')
                                plt.figtext(0.65, 0.65, 'Unaveraged Max: ' +
                                            str(unavg_max_bool))
                                plt.figtext(0.65, 0.55, 'Slope: ' +
                                            str(round(a_l, 7)) + ',' + str(round(a_r, 7)))
                            plt.xlabel('E')
                            plt.ylabel('Probability')
                            plt.title('E product')
                            plt.legend()
                            plt.tight_layout()
                            pdf.savefig()
                            pdfE.savefig()
                            plt.close()

                        # if external_plotting:
                        #     with open("./inputs/pe" + str(i+1) + "_QCT.txt", "w") as txt_file:
                        #         for j in range(len(E)):
                        #             txt_file.write(str(E[j]) + ' ' + str(p[j]) + '\n')
                        #
                        #     with open("./inputs/pe" + str(i+1) + "_Grid.txt", "w") as txt_file:
                        #         for j in range(len(grid)):
                        #             txt_file.write(str(grid[j]) + ' ' + str(p_grid[j]) + '\n')

                    if filename == 'pv':

                        grid = vgridout

                        p_full_avg = []
                        for j in range(len(p)):
                            if j-1 < 0 or j+1 > (len(p)-1):
                                p_full_avg.append(p[j])
                            else:
                                p_full_avg.append(np.mean(p[j-1:j+1+1]))

                        p_on_grid = []
                        p_on_grid_avg = []
                        for ele in grid:
                            p_on_grid.append(p[list(E).index(round(ele, 1))])
                            p_on_grid_avg.append(p_full_avg[list(E).index(round(ele, 1))])

                        index_of_max = np.argmax(np.delete(np.array(p_on_grid), 0))+1
                        num_nb = 3

                        if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                            unavg_max_bool, a_l, b_l, a_r, b_r = unavg_max(
                                p_on_grid_avg[index_of_max-num_nb:index_of_max+num_nb+1], grid[index_of_max-num_nb:index_of_max+num_nb+1], num_nb, filename, low_prob)
                        else:
                            unavg_max_bool = True
                            a_l = 0
                            b_l = 0
                            a_r = 0
                            b_r = 0

                        p_on_grid_final = []
                        for j in range(len(grid)):
                            if ((index_of_max-1 <= j <= index_of_max+1) and unavg_max_bool) or (0 <= j <= 4):
                                if j == index_of_max:
                                    p_on_grid_final.append(p_on_grid[j])
                                else:
                                    p_on_grid_final.append(p_on_grid_avg[j])
                            else:
                                p_on_grid_final.append(p_on_grid_avg[j])

                        for ele in p_on_grid_final:
                            txt_file.write(str(ele) + ',')

                        if plotting:

                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                xrange = np.linspace(grid[index_of_max-2]-3,
                                                     grid[index_of_max+2]+3, 50)

                            plt.figure()
                            plt.plot(E, p, '-b', label='QCT')
                            plt.plot(grid, p_on_grid_final, '.r', label='Grid')
                            plt.plot(grid[index_of_max], p_on_grid[index_of_max], '.g')
                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                plt.plot(xrange, line(xrange, a_l, b_l), '-k', label='Line1')
                                plt.plot(xrange, line(xrange, a_r, b_r), '-k', label='Line2')
                                plt.figtext(0.65, 0.65, 'Unaveraged Max: ' +
                                            str(unavg_max_bool))
                                plt.figtext(0.65, 0.55, 'Slope: ' +
                                            str(round(a_l, 7)) + ',' + str(round(a_r, 7)))
                            plt.xlabel('v')
                            plt.ylabel('Probability')
                            plt.title('v product')
                            plt.legend()
                            plt.tight_layout()
                            pdf.savefig()
                            pdfv.savefig()
                            plt.close()

                    if filename == 'pj':

                        grid = jgridout

                        p_full_avg = []
                        for j in range(len(p)):
                            if j-1 < 0 or j+1 > (len(p)-1):
                                p_full_avg.append(p[j])
                            elif j-2 < 0 or j+2 > (len(p)-2):
                                p_full_avg.append(np.mean(p[j-1:j+1+1]))
                            elif j-3 < 0 or j+3 > (len(p)-3):
                                p_full_avg.append(np.mean(p[j-2:j+2+1]))
                            elif j-4 < 0 or j+4 > (len(p)-4):
                                p_full_avg.append(np.mean(p[j-3:j+3+1]))
                            elif j-5 < 0 or j+5 > (len(p)-5):
                                p_full_avg.append(np.mean(p[j-4:j+4+1]))
                            elif j-6 < 0 or j+6 > (len(p)-6):
                                p_full_avg.append(np.mean(p[j-5:j+5+1]))
                            elif j-7 < 0 or j+7 > (len(p)-7):
                                p_full_avg.append(np.mean(p[j-6:j+6+1]))
                            else:
                                p_full_avg.append(np.mean(p[j-7:j+7+1]))

                        p_on_grid = []
                        p_on_grid_avg = []

                        for ele in grid:
                            p_on_grid.append(p[list(E).index(round(ele, 1))])
                            p_on_grid_avg.append(p_full_avg[list(E).index(round(ele, 1))])

                        index_of_max = np.argmax(np.array(p_on_grid))
                        num_nb = 3

                        if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                            unavg_max_bool, a_l, b_l, a_r, b_r = unavg_max(
                                p_on_grid_avg[index_of_max-num_nb:index_of_max+num_nb+1], grid[index_of_max-num_nb:index_of_max+num_nb+1], num_nb, filename, low_prob)
                        else:
                            unavg_max_bool = False
                            a_l = 0
                            b_l = 0
                            a_r = 0
                            b_r = 0

                        p_on_grid_final = []
                        for j in range(len(grid)):
                            if (index_of_max-2 <= j <= index_of_max+2) and unavg_max_bool:
                                if j == index_of_max:
                                    p_on_grid_final.append(p_on_grid[j])
                                else:
                                    p_on_grid_final.append(p_on_grid_avg[j])
                            else:
                                p_on_grid_final.append(p_on_grid_avg[j])

                        for ele in p_on_grid_final:
                            txt_file.write(str(ele) + ',')

                        if plotting:

                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                xrange = np.linspace(grid[index_of_max-2]-5,
                                                     grid[index_of_max+2]+5, 50)

                            plt.figure()
                            plt.plot(E, p, '-b', label='QCT')
                            plt.plot(grid, p_on_grid_final, '.r', label='Grid')
                            plt.plot(grid[index_of_max], p_on_grid[index_of_max], '.g')
                            if index_of_max-num_nb >= 0 and index_of_max+num_nb <= len(p_on_grid)-1:
                                plt.plot(xrange, line(xrange, a_l, b_l), '-k', label='Line1')
                                plt.plot(xrange, line(xrange, a_r, b_r), '-k', label='Line2')
                                plt.figtext(0.65, 0.65, 'Unaveraged Max: ' +
                                            str(unavg_max_bool))
                                plt.figtext(0.65, 0.55, 'Slope: ' +
                                            str(round(a_l, 7)) + ',' + str(round(a_r, 7)))
                            plt.xlabel('j')
                            plt.ylabel('Probability')
                            plt.title('j product')
                            plt.legend()
                            plt.tight_layout()
                            pdf.savefig()
                            pdfj.savefig()
                            plt.close()

                txt_file.write('\n')

if plotting:
    pdf.close()
    pdfE.close()
    pdfv.close()
    pdfj.close()

print('Number of features: ' + str(num_features))
print('Number of outputs: ' + str(num_outputs))
