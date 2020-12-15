import matplotlib.pyplot as plt
import scipy.stats as stats

## Data /s, min from 3 runs.
processes = [2,4,8,12,16]
processes_2 = [2,4,8, 16]
ex_2_1 = [6.162598, 3.207325, 1.429123, 0.966994, 0.790304]
ex_2_2 = [5.566579, 2.921690, 1.453972, 0.734841]
ex_2_3 = [5.544093, 2.883076, 1.448170, 0.976962, 0.730453]
ex_2_4 = [5.336779, 2.769113, 1.389383, 0.932238, 0.702393]
ex_2_5 = [5.432652, 2.830847, 1.417490, 0.951843, 0.717498]
ex_2_6 = [5.293030, 2.759023, 1.406318, 0.954788, 0.740413]

plt.plot(processes, ex_2_1, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes_2, ex_2_2, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_3, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_4, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_5, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_6, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_1, linestyle='--', marker='o', label="1.2.1")
plt.plot(processes_2, ex_2_2, linestyle='--', marker='o', label="1.2.2")
plt.plot(processes, ex_2_3, linestyle='--', marker='o', label="1.2.3")
plt.plot(processes, ex_2_4, linestyle='--', marker='o', label="1.2.4")
plt.plot(processes, ex_2_5, linestyle='--', marker='o', label="1.2.5")
plt.plot(processes, ex_2_6, linestyle='--', marker='o', label="1.2.6")
plt.legend()
plt.show()

ex_3_size = [2**x for x in range(3, 3+28)]
# ex_3_intra_time = [0.000000398, 0.000000253, 0.000000262, 0.000000257, 0.000000306, 0.000000321, 0.000000477, 0.000000527, 0.000000613, 0.000002057, 0.000002612, 0.000003417, 0.000004582, 0.000006977, 0.000011977, 0.000020714, 0.000039031, 0.000089573, 0.000301326, 0.000760094, 0.00239259, 0.0033164, 0.009278028, 0.013361149, 0.029216746, 0.049635408, 0.084598251, 0.167728959]
# ex_3_inter_time = [0.000000373, 0.000000401, 0.000000265, 0.000000266, 0.000000445, 0.000000326, 0.000000635, 0.000000675, 0.000000892, 0.000002142, 0.000002687, 0.000003457, 0.00000473, 0.000006659, 0.000011208, 0.000020826, 0.000038234, 0.00007388, 0.000154122, 0.000497825, 0.001398401, 0.002799326, 0.005522974, 0.010739649, 0.020810559, 0.040751144, 0.079919851, 0.161277779]
#ex_3_intra_time = [0.000000446, 0.000000295, 0.000000302, 0.000000306, 0.000000355, 0.000000373, 0.000000588, 0.000000636, 0.000000756, 0.000002423, 0.000003055, 0.000003974, 0.000005336, 0.000008059, 0.000013336, 0.000024314, 0.000045373, 0.000081852, 0.000155939, 0.000492048, 0.001305051, 0.002673494, 0.00539996, 0.010787497, 0.021972588, 0.042552521, 0.082991089, 0.168396058]
#ex_3_intra_time = [0.000000380, 0.000000241, 0.000000499, 0.000000523, 0.000000566, 0.000000580, 0.000000978, 0.000001039, 0.000000758, 0.000002077, 0.000002594, 0.000003388, 0.000004440, 0.000007027, 0.000011263, 0.000020580, 0.000158472, 0.000074146, 0.000269741, 0.000946442, 0.002312403, 0.004799637, 0.009040121, 0.012200211, 0.030033639, 0.060959254, 0.121439994, 0.242536179]
ex_3_intra_time = [0.000000406, 0.000000244, 0.000000523, 0.000000402, 0.000000564, 0.000000592, 0.000000597, 0.000000680, 0.000000748, 0.000002139, 0.000002667, 0.000003466, 0.000004396, 0.000006627, 0.000011114, 0.000020417, 0.000158795, 0.000074688, 0.000316531, 0.001157599, 0.002631885, 0.003857160, 0.005373242, 0.010553650, 0.020727785, 0.040637550, 0.080326486, 0.159385863]
ex_3_inter_time = [0.000053885, 0.000053324, 0.000053329, 0.00005322, 0.000053068, 0.000052929, 0.000052826, 0.000061053, 0.000092799, 0.000092487, 0.00017566, 0.000208156, 0.000386743, 0.000754518, 0.001328073, 0.002558548, 0.004996472, 0.010817858, 0.023895512, 0.047600412, 0.072726169, 0.144779066, 0.29132094, 0.589569401, 1.257257961, 2.353205263, 4.648220828, 9.279981464]

# Do linear regression on the ex 3 data, inverse of slope = bandwidth,
# intercept = latency.
intra_slope, intra_intercept, _, _, _ = stats.linregress(ex_3_size, ex_3_intra_time)

inter_slope, inter_intercept, _, _, _ = stats.linregress(ex_3_size, ex_3_inter_time)

print("EX 3: intra: BANDWIDTH = " + str((1 / intra_slope) / 1000000000)+ " GB/s")
print("EX 3: intra: LATENCY  = " + str(intra_intercept))
print("EX 3: inter: BANDWIDTH = " + str((1 / inter_slope)/1000000000) + " GB/s")
print("EX 3: inter: LATENCY  = " + str(inter_intercept))
plt.plot(ex_3_size, ex_3_intra_time)
plt.xlabel("Message size")
plt.ylabel("Time sec")
plt.show()
plt.plot(ex_3_size, ex_3_inter_time)
plt.xlabel("Message size")
plt.ylabel("Time sec")
plt.show()