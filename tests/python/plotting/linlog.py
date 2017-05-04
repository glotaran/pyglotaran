import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Numbers from -50 to 50, with 0.1 as step
xdomain = np.arange(-50,50, 0.1)

axMain = plt.subplot(111)
axMain.plot(np.sin(xdomain), xdomain)
axMain.set_xscale('linear')
axMain.set_xlim((0.5, 1.5))
axMain.spines['left'].set_visible(False)
axMain.yaxis.set_ticks_position('right')
axMain.yaxis.set_visible(False)


divider = make_axes_locatable(axMain)
axLin = divider.append_axes("left", size=2.0, pad=0, sharey=axMain)
axLin.set_xscale('log')
axLin.set_xlim((0.01, 0.5))
axLin.plot(np.sin(xdomain), xdomain)
axLin.spines['right'].set_visible(False)
axLin.yaxis.set_ticks_position('left')
#plt.setp(axLin.get_xticklabels(), visible=True)

plt.title('Linear right, log left')
plt.show()