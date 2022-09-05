import matplotlib.pyplot as plt
import re
import sys

steps = []
losses = []
test_set_losses = []
with open(sys.argv[1]) as f:
    for line in f:
        m = re.search(
            'step ([0-9]+)\. loss: ([0-9]+.[0-9]+), test_set_loss: ([0-9]+.[0-9]+)', line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            test_set_losses.append(float(m.group(3)))

plt.plot(steps, losses, label='losses')
plt.plot(steps, test_set_losses, label='test_set_losses')
plt.legend()
plt.show()
