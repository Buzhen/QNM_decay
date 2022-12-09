import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print('Plotting Data...')

fig, ax = plt.subplots()
ax.set_xlabel('x label')
ax.set_ylabel('y label')
i=[0]

def f1():
    ax.plot([1, 5, 10, 20], [1, 5, 10, 20])

def f2():
    ax.plot([1, 5, 10, 20], [2, 10, 20, 40])

def f3():
    ax.plot([1, 5, 10, 20], [5, 9, 17, 28])


def update(event=None):
    if i[0]==0: f1()
    if i[0]==1: f2()
    if i[0]==2: f3()
    fig.canvas.draw_idle()
    i[0]+=1
    print('Step {} done..., Press a key to continue'.format(i[0]))

fig.canvas.mpl_connect("key_press_event", update)

update()
plt.show()