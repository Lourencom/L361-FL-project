import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt


B_simples = [[-11687.922491437468, 50476.83433348289, -8360.412075212149, 17319.119343169154, -25954.21660271833, 23370.69120739089, -4962.304179443327, -171059.17398979756], [-19385.786387690354, -48296.90692407105, 14661.483253924791, -13137.487134913083, -78752.06911183994, -43549.34807501476, 33191.38345639109, -8916.155326480151, 26379.271823599152, -17644.262975900085, 56942.85065950044], [63790.0779072847, 18194.037770084386, 20838.652107947135, 67274.89421292844, -313061.2900299799, -271432.3756642664, -63409.341255032, 100693.78565294151, -405953.26435536693, -57494.052391349855, -20407.116358130268], [136404.8146034731, 382788.4590252541, -303602.1198341131, -391914.46840772225, -100359.90833068614, -222389.64401717987, -67227.1535142801, 99115.03185441485, -141411.7204019892, 67098.77896076355, -87909.66401102718, 673764.565140841, -89414.50505702938, -128119.07932215642, -81017.08955323228, 168047.6320547225, -570472.3537002297, -27778430.098110802, 228947.00354616542, -91905.22217824718, -143878.5904577031, 428230.9802809606, -65157.41155478921, -81947.86215890732, -50379.05382373369, -114053.67299065265, -100041.06682634717, -62604.96306232306, -257614.42012444683, -47455.47731560237, 227508.80388985266, 120620.7368087462, -68318.8873917895, -85052.34328022464, 92665.1515040749, -324533.92759395094, -456602.2352201042, -886315.7575465435, 793226.2717526885, 346927.0548617065, -219646.0704645006, 409218.6427205505, 312957.09194129705, -217288.8919829346, -230479.10911932818, -141406.95367459086, 188017.85735417364]]
B_simples = [[abs(el) for el in subl] for subl in B_simples]
# apply median filter for each sublist
B_simple_median = [medfilt(subl, 5) for subl in B_simples]

rounds = 10

mean_t = [np.mean([B_simples[run][i] for run in range(len(B_simples)) if i < len(B_simples[run])]) for i in range(rounds)]
std_t = [np.std([B_simples[run][i] for run in range(len(B_simples)) if i < len(B_simples[run])]) for i in range(rounds)]

mean_t = np.array(mean_t)

print(np.mean(mean_t)) # 108600

print(np.mean(std_t))
std_t = np.array(std_t)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_t, color='darkblue')
ax.fill_between(range(rounds), mean_t - std_t, mean_t + std_t, color='lightblue', alpha=0.5)
ax.set_xlabel('Rounds')
ax.set_ylabel('Estimation of Critical batch size')
ax.set_title(f'Estimation of Critical batch size for each round (mean = {np.mean(mean_t):.0f}, std = {np.mean(std_t):.0f})')
plt.show()
out_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'critical_bs_estimation.png'), dpi=300)



# plot the critical bs estimation for iid

B_simples = [[-18407.686925566362, -13762.505288528548, -29665.108319094634, -12927.35371816805, 4372.760959719874, 10805.953609141283, -21040.561981702886, 23238.6966912209, -20496.066521615783, 40973.22031090615], [-40754.83941407597, -94828.01133622219, -168606.6061223791, -13053.941180392021, -12196.393251686653, 139131.94393159807, -18836.993638551605, 14911.932928595123, 71887.69261004392, -429673.47555676906], [-50326.972540666844, -221078.50863027747, -12331.574093571728, -10559.19947656049, -533488.2914706912, 72097.91998244265, 2756775.10465039, -37545.04286075924, -215591.0786897387, -124943.4867790073], [-86654.8377399388, 70070.96612662714, 869667.1183916295, 62085.1567899368, 47205.20980176431, -64801.69123063742, 59619.394715874994, 43996.21136494777, 76026.20309461898, 370612.13195318024], [-37359.721294111616, -179548.41406089318, -266947.3063659719, -956503.1802564623, 272930.986020992, -264428.3445225671, 100874.68615423376, -94089.052502197, -138475.6462939103, -258445.3923542993]]
B_simples = [[abs(el) for el in subl] for subl in B_simples]
# apply median filter for each sublist
B_simple_median = [medfilt(subl, 5) for subl in B_simples]

rounds = 10

mean_t = [np.mean([B_simples[run][i] for run in range(len(B_simples)) if i < len(B_simples[run])]) for i in range(rounds)]
std_t = [np.std([B_simples[run][i] for run in range(len(B_simples)) if i < len(B_simples[run])]) for i in range(rounds)]

mean_t = np.array(mean_t)

print(np.mean(mean_t)) # 191000
std_t = np.array(std_t)
print(np.mean(std_t)) 

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_t, color='darkblue')
ax.fill_between(range(rounds), mean_t - std_t, mean_t + std_t, color='lightblue', alpha=0.5)
ax.set_xlabel('Rounds')
ax.set_ylabel('Estimation of Critical batch size')
ax.set_title(f'Estimation of Critical batch size for each round (mean = {np.mean(mean_t):.0f}, std = {np.mean(std_t):.0f})')
# set log y scale


plt.show()
out_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'critical_bs_estimation_iid.png'), dpi=300)

