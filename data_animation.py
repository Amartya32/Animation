import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg') # Need to use in order to run on mac
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

#=============================================================================================

t_start = 200# start frame
t_end = 1841# end frame

data = pd.read_csv('kinematicgrasp.csv') # only coordinate data
df = data.loc[t_start:t_end,'WRUX':'FT5Z']

# Find max and min values for animation ranges
df_minmax = pd.DataFrame(index=list('xyz'),columns=range(2))
print(df_minmax)
for i in list('xyz'):
    c_max = df.filter(regex='_{}'.format(i)).max().max()
    c_min = df.filter(regex='_{}'.format(i)).min().min()
    df_minmax.loc[i] = np.array([c_min,c_max])

df_minmax = 3.3*df_minmax # increase by 30% to make animation look better

df.columns  = np.repeat(range(26),3) # store cols like this for simplicity
print(df.columns)
N_tag = int(df.shape[1]/3) # nr of tags used (all)
print(N_tag)
N_trajectories = N_tag

t = np.linspace(0,data.Time[t_end],df.shape[0]) # pseudo time-vector for first walking activity
x_t = np.zeros(shape=(N_tag,df.shape[0],3)) # empty animation array (3D)

for tag in range(26):
 #     # store data in numpy 3D array: (tag,time-stamp,xyz-coordinates)
    x_t[tag,:,:] = df[tag]

x_t = x_t[:, :, [0, 2, 1]]
print(x_t.shape)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.view_init(10, 3)
pts = []
for i in range(x_t.shape[0]):
    pts += ax.plot([], [], [], 'o')

ax.set_xlim3d([np.nanmin(x_t[:, :, 0]), np.nanmax(x_t[:, :, 0])])
ax.set_ylim3d([np.nanmin(x_t[:, :, 1])-400, np.nanmax(x_t[:, :, 1])+400])
ax.set_zlim3d([np.nanmin(x_t[:, :, 2]), np.nanmax(x_t[:, :, 2])])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

# animation function
def animate(i):
    for pt, xi in zip(pts, x_t):
        x, y, z = xi[:i].T
        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])   
    return pts

# Animation object
anim = animation.FuncAnimation(fig, func=animate, frames=x_t.shape[1], interval=1000/150, blit=True)

plt.show()
# # choose a different color for each trajectory
# colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))
# # set up trajectory lines
# lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
# # set up points
# pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])
# # set up lines which create the stick figures

# stick_defines = [
#     (0, 1),
#     (1, 2),
#     (3, 4),
#     (4, 5),
#     (6, 7),
#     (7, 8),
#     (9, 10),
#     (10, 11)
# ]

#stick_lines = [ax.plot([], [], [], 'k-')[0] for _ in stick_defines]

#prepare the axes limits
# ax.set_xlim(df_minmax.loc['x'].values)
# ax.set_ylim(df_minmax.loc['z'].values) # note usage of z coordinate
# ax.set_zlim(df_minmax.loc['y'].values) # note usage of y coordinate

# set point-of-view: specified by (altitude degrees, azimuth degrees)
# ax.view_init(10, 150)
# # initialization function: plot the background of each frame
# pts = []

# for i in range(df_minmax.shape[0]):
#     pts += ax.plot([], [], [], 'o')





# # # animation function.  This will be called sequentially with the frame number
# def animate(i):
#     # we'll step two time-steps per frame.  This leads to nice results.
#     #i = (5 * i) % x_t.shape[1]

#     for pt, xi in zip(pts, x_t):
#         x, y, z = xi[:i].T # note ordering of points to line up with true exogenous registration (x,z,y)
#         pt.set_data(x[-1:], y[-1:])
#         pt.set_3d_properties(z[-1:])

#     # for stick_line, (sp, ep) in zip(stick_lines, stick_defines):
#     #     stick_line._verts3d = x_t[[sp,ep], i, :].T.tolist()

#     #ax.view_init(300, 0.3 * i)
#     fig.canvas.draw()
#     return pts

# # instantiate the animator.
# anim = animation.FuncAnimation(fig, animate, frames=1842, interval=50, blit=True)

# plt.show()