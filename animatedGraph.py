# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:30:03 2020

@author: Anup
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


indep=X[:,1]

fig, ax = plt.subplots()
line, = ax.plot(indep, Pre, color='k')



def update(num, x, y, line):
    line.set_data(indep[:num],Pre[:num] )
    line.axes.axis([2500, 11000, 350, 2000])
    return line,

ani = animation.FuncAnimation(fig, update, len(X[:,1]), fargs=[X[:,1],regressor_OLS.predict(X[:,1]) , line],
                              interval=500, blit=False)

ani.save('test.mp4')
plt.show()