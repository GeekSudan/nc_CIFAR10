import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

znd = pd.read_table('znd1.txt')
znd_train_loss = znd["Train Loss"]
znd_test_loss = znd["Valid Loss"]
znd_train_accuracy = znd["Train Acc."]
znd_test_accuracy = znd["Valid Acc."]

momentum = pd.read_table('momentum.txt')
mom_train_loss = momentum["Train Loss"]
mom_test_loss = momentum["Valid Loss"]
mom_train_accuracy = momentum["Train Acc."]
mom_test_accuracy = momentum["Valid Acc."]

adam = pd.read_table('adam.txt')
adam_train_loss = adam["Train Loss"]
adam_test_loss = adam["Valid Loss"]
adam_train_accuracy = adam["Train Acc."]
adam_test_accuracy = adam["Valid Acc."]

plt.figure(figsize=(15, 10))
plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None, wspace=None, hspace=None)

plt.subplot(2, 2, 1)
np.save('train_loss', znd_train_loss, mom_train_loss, adam_train_loss)
plt.plot(list(range(100)), znd_train_loss, color='orangered', linestyle='--', label='ZND')
plt.plot(list(range(100)), mom_train_loss, color='limegreen', linestyle='dotted', label='SGD-M')
plt.plot(list(range(100)), adam_train_loss, color='cornflowerblue', linestyle='dashdot', label='ADAM')
nn_plot = np.arange(0, 120, 20)
nn_plot2 = np.arange(0, 2.0, 0.2)
plt.xticks(nn_plot)
plt.yticks(nn_plot2)
plt.ylim(0, 2.0)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Train Loss", fontsize=12)
plt.savefig("train_loss.eps", format='eps', dpi=1000)
plt.legend()
# plt.show()

plt.subplot(2, 2, 2)
np.save('test_loss', znd_test_loss, mom_test_loss, adam_test_loss)
plt.plot(list(range(100)), znd_test_loss, 'orangered', linestyle='--', label='ZND')
plt.plot(list(range(100)), mom_test_loss, 'limegreen', linestyle='dotted', label='SGD-M')
plt.plot(list(range(100)), adam_test_loss, 'cornflowerblue', linestyle='dashdot', label='ADAM')
nn_plot = np.arange(0, 120, 20)
nn_plot2 = np.arange(0, 2.0, 0.2)
plt.xticks(nn_plot)
plt.yticks(nn_plot2)
plt.ylim(0, 2.0)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Valid Loss", fontsize=12)
plt.savefig("test_loss.eps", format='eps', dpi=1000)
plt.legend()
# plt.show()


plt.subplot(2, 2, 3)
np.save('train_accuracy', znd_train_accuracy, mom_train_accuracy, adam_train_accuracy)
plt.plot(list(range(100)), znd_train_accuracy, 'orangered', linestyle='--', label='ZND')
plt.plot(list(range(100)), mom_train_accuracy, 'limegreen', linestyle='dotted', label='SGD-M')
plt.plot(list(range(100)), adam_train_accuracy, 'cornflowerblue', linestyle='dashdot', label='ADAM')
nn_plot = np.arange(0, 120, 20)
nn_plot2 = np.arange(50, 120, 10)
plt.xticks(nn_plot)
plt.yticks(nn_plot2, ('0', '50', '60', '70', '80', '90',  '100'))
# plt.ylim(50, 100)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Train Accuracy", fontsize=12)
plt.savefig("train_accuracy.eps", format='eps', dpi=1000)
plt.legend()
# plt.show()

plt.subplot(2, 2, 4)
np.save('test_accuracy', znd_test_accuracy, mom_test_accuracy, adam_test_accuracy)
plt.plot(list(range(100)), znd_test_accuracy, 'orangered', linestyle='--', label='ZND')
plt.plot(list(range(100)), mom_test_accuracy, 'limegreen', linestyle='dotted', label='SGD-M')
plt.plot(list(range(100)), adam_test_accuracy, 'cornflowerblue', linestyle='dashdot', label='ADAM')
nn_plot = np.arange(0, 120, 20)
nn_plot2 = np.arange(50, 120, 10)
plt.xticks(nn_plot)
plt.yticks(nn_plot2, ('0', '50', '60', '70', '80', '90',  '100'))
# plt.ylim(50, 100)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Valid Accuracy", fontsize=12)
plt.savefig("test_accuracy.eps", format='eps', dpi=1000)
plt.legend()

plt.show()
