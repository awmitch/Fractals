import numpy as np
import pylab
import matplotlib.pyplot as plt
from Tkinter import *
import matplotlib
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from numba import jit
import matplotlib.animation as animation
from random import randint,shuffle
#import cmath
import time
import glob
import cmath

@jit
def mandel(x, y, max_iters,eqn_list,d):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x,y)
    z = 0j
    
    for i in range(max_iters):
        for eqn in eqn_list:
            if eqn == 0:
                z = z**d+c
            elif eqn == 1:
                d = randint(1,3)
                z = z**d+c
            elif eqn == 2:
                z = ((abs(z.real)+abs(z.imag)*1j)**2) + c
            elif eqn == 3:
                z = cmath.exp(z) + c
            elif eqn == 4:
                z = cmath.cos(z) +c
            elif eqn == 5:
                z = c*z*(1-z)
        if z.real * z.real + z.imag * z.imag >= 4:
            return 255 * (i+1-(np.log(2)/abs(z))/np.log(2)) // max_iters
#    else:
#    for i in range(max_iters):
#        z = (z*z)+c
#         = cmath.cos(z)*cmath.cos(z) + c
    #        z = (z*z*z)+c
#            z = ((abs(z.real)+abs(z.imag)*1j)**2.0) + c  
    #        z = z*(z+chan) + c
    #        z = z*(z-2) + c
    #        z = z*(z+2) + c
            
    #        z = (4.0*(z*z-z+1.0)**3.0)/(27.0*z*z*(z-1.0)*(z-1.0))
    #        z = z*z+1j+c
    #        z = z*(z-2.0) + c
                  
    return 255

@jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters, eqn_list,d):
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters,eqn_list,d)
            image[y, x] = color

    return image

class App:
    def __init__(self,master):
        self.master = master
        self._job = None
        self.frame = Frame(self.master)
        self.menubar = Menu(self.master)
        master.config(menu=self.menubar)
        self.menubar.add_command(label="Refresh!", command=self.graph_update)
        self.flag_menu = Menu(self.menubar)
        self.menubar.add_cascade(label="Coloring",menu=self.flag_menu)
        self.frame.grid(row=0,column=0,sticky=W)
        self.low_frame = Frame(self.master)
        self.low_frame.grid(row=1,column=0,sticky=W)
        self.nav_frame = Frame(self.low_frame)
        self.nav_frame.grid(row=0,column=9,sticky=E)
        self.Fig = matplotlib.figure.Figure(figsize=(self.master.winfo_screenwidth()*19/1920.0,self.master.winfo_screenheight()*9.5/1080),dpi=100,tight_layout=True,frameon=False)
        self.FigSubPlot = self.Fig.add_subplot(111)    
        self.canvas = FigureCanvasTkAgg(self.Fig, master=self.frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(fill=BOTH)
        self.toolbar = NavigationToolbar(self.canvas, self.nav_frame)
        self.zoom_flag = 0
        self.iter_flag = 0
        self.chan_flag = 0
        
        self.fun_var = StringVar()
        self.fun_var.set("z*z+c")
        self.iter_init = 50
        self.iter_target = 550
        self.frame_num = 500
        self.iter = self.iter_init
        
        self.eqn_list = [0]
        self.d = 2
        self.eqn_list_str = ["z**%s+c"%self.d,"z**d+c","(|z.real|+|z.imag|*i)**2 + c","exp(z) + c","cos(z) + c","c*z*(1-z)"]

        
        self.iter_var = IntVar()
        self.iter_var.set(200)
        self.menubar.add_command(label=self.iter_var.get())
        self.menubar.add_command(label="Options",command=self.options)
        self.zoom_menu = Menu(self.menubar)
        self.menubar.add_cascade(label="Zoom",menu=self.zoom_menu)
        self.zoom_menu.add_command(label='Animate',command=self.zoom)
        self.radio_zoom_var = StringVar()
        self.radio_zoom_var.set('Const Iter')
        self.zoom_menu.add_radiobutton(label="Const Iter", variable=self.radio_zoom_var)
        self.zoom_menu.add_radiobutton(label="Var Iter", variable=self.radio_zoom_var)
        self.iter_menu = Menu(self.menubar)
        self.menubar.add_cascade(label="Iterate",menu=self.iter_menu)
        self.iter_menu.add_command(label='Animate',command=self.iterate)
        self.chan_menu = Menu(self.menubar)
        self.menubar.add_cascade(label="Var Fun",menu=self.chan_menu)
        self.chan_menu.add_command(label='Animate',command=self.change)
        self.menubar.add_command(label="Randomize",command=self.init_rand)
        
#        self.xy,self.xrange = (-0.77935557422765123, -0.13446256411143759),[-0.77935704110987425, -0.77935482514883669]
#        self.xy,self.xrange = (-0.75222388519157268, -0.040965152572997689),[-0.75222388519159378, -0.75222388519155148]
#        self.xy,self.xrange = [-0.10228392078305387, -0.10228392078297188],(-0.1022839207830129, -0.94626442954727208)
#        self.xy,self.xrange = (-0.90788399028954736, -0.26752268187573769),[-0.90788399028955979, -0.90788399028953526]
#        self.iter_var.trace('w',self.slider_iter)
#        self.xy,self.xrange =(-1.1476766998672108, 0.27721359784019722),[-1.1476766998672145, -1.1476766998672074]
#        self.xy,self.xrange = (0.92210302746896, -0.069229029462414271),[-2.0, 1.0]
        self.xy,self.xrange = (0.74781121734121658, -0.12741123251340625),[-0.67097267147858242, 2.3290273285214176]
        self.prev_iter = None
        self.color_var = StringVar()
        self.color_var.set('flag')
        self.color_var.trace('w',self.graph_update)
        for string in ['viridis','inferno','plasma','magma','Blues','BuGn','BuPu','GnBu','Greens','Greys','Oranges','OrRd','PuBu','PuBuGn','PuRd','Purples','RdPu','Reds','YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot','autumn','bone','cool','copper','gist_heat','gray','hot','pink','spring','summer','winter','BrBG','bwr','coolwarm','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu','RdYlGn','Spectral','seismic','Accent','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','gist_earth','terrain','ocean','gist_stern','brg','CMRmap','cubehelix','gnuplot','gnuplot2','gist_ncar','nipy_spectral','jet','rainbow','gist_rainbow','hsv','flag','prism']:
            self.flag_menu.add_radiobutton(label=string, variable=self.color_var)

        self.xlim = [-2.0,1.0]
        self.ylim = [-1.0,1.0]

        image = np.zeros((1000, 2000), dtype=np.uint8)
        create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get(),[0],2)
#        self.FigSubPlot.axis('off')
        
        self.im = self.FigSubPlot.imshow(image,cmap=self.color_var.get(),animated=True,interpolation='none')
        self.canvas.draw()
        self.rxlim = [self.FigSubPlot.get_xlim()[0],self.FigSubPlot.get_xlim()[1]]
        self.rylim = [self.FigSubPlot.get_ylim()[0],self.FigSubPlot.get_ylim()[1]]
        self.master.bind("<ButtonRelease-1>",self.pan_update)
        self.master.bind("<ButtonRelease-3>",self.pan_update)
        self.master.bind("<MouseWheel>",self._on_mousewheel)
        self.canvas.callbacks.connect('button_press_event',self.set_coords)
    def init_rand(self):
        self.time = time.time()
        self.rand_flag = 1
        self.randomize()
        self.options()
    def check_rand(self):
        if self.rand_flag == 1:
            self.randomize()
            self.window.destroy()
            self.eqn_list_str = ["z**%s+c"%self.d,"z**d+c","(|z.real|+|z.imag|*i)**2 + c","exp(z) + c","cos(z) + c","c*z*(1-z)"]
            self.options()
    def randomize(self):

        l = ['viridis','inferno','plasma','magma','Blues','BuGn','BuPu','GnBu','Greens','Greys','Oranges','OrRd','PuBu','PuBuGn','PuRd','Purples','RdPu','Reds','YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot','autumn','bone','cool','copper','gist_heat','gray','hot','pink','spring','summer','winter','BrBG','bwr','coolwarm','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu','RdYlGn','Spectral','seismic','Accent','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','gist_earth','terrain','ocean','gist_stern','brg','CMRmap','cubehelix','gnuplot','gnuplot2','gist_ncar','nipy_spectral','jet','rainbow','gist_rainbow','hsv','flag','prism']
        self.num_eqn = randint(2,3)
        self.eqn_list = []
        for it in range(0,self.num_eqn):
            eqn = randint(0,5)
            while eqn in self.eqn_list:
                eqn = randint(0,5)
            self.eqn_list.append(eqn)
        print self.eqn_list
        image = np.zeros((1000, 2000), dtype=np.uint8)
        self.iter_var.set(100)
        self.d = randint(2,4)
        create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image,self.iter_var.get(),self.eqn_list,self.d)
        self.im = self.FigSubPlot.imshow(image,cmap=l[randint(0,len(l)-1)])
        self.canvas.show()
        print self.time-time.time()
        self.time = time.time()
        self.master.after(300,self.check_rand)
        
    def set_coords(self,event):
        self.rand_flag = 0
        self.coords = ((event.xdata*(self.xlim[1]-self.xlim[0])/(self.rxlim[1]-self.rxlim[0]))+self.xlim[0],self.ylim[0]-(event.ydata*(self.ylim[1]-self.ylim[0])/(self.rylim[1]-self.rylim[0])))
        print "coords",self.coords
        print "xlim",self.xlim
    def zoom(self):
        self.zoom_flag = 1
        if self.radio_zoom_var.get() == "Const Iter":
            self.const_flag = 0
        else:
            self.const_flag = 1
        self.animation()
        self.zoom_flag = 0
    def iterate(self):
        self.iter_flag = 1
        self.animation()
        self.iter_flag = 0
    def change(self):
        self.chan_flag = 1
        self.animation()
        self.chan_flag = 0
    def options(self):
        self.window = Toplevel(self.master)
        self.window.geometry("%dx%d%+d%+d" % (200, 70, 50, 50))
        self.window.lift()
        frame = Frame(self.window)
        self.eqn_listbox = Listbox(frame)
        self.eqn_listbox.pack(fill=BOTH)
        for eqn in self.eqn_list:
            self.eqn_listbox.insert(END,self.eqn_list_str[eqn])
        frame.pack(fill=BOTH)
        
    def animation(self):
        image = np.zeros((1000, 2000), dtype=np.uint8)
#        self.pointx,self.pointy=-0.73295433265572163, 0.24086484682261031
#        self.pointx,self.pointy=0.37213771619186042, 0.090398260468917052
#        self.pointx = 0.001643721971153
#        self.pointy = -0.822467633298876
#        self.pointx = -0.77568377
#        self.pointy = 0.13646737
#        self.pointx = -0.3905407802
#        self.pointy = -0.5867879073

        self.pointx,self.pointy=self.xy[0],self.xy[1]
        self.tolerance = -self.xrange[0]+self.xrange[1]
        print self.tolerance
        self.ytolerance = self.tolerance*(2.0/3.0)
        self.xlim = [-(self.tolerance/2.0)+self.pointx, (self.tolerance/2.0)+self.pointx]
        self.ylim = [self.pointy-self.tolerance*(2.0/3.0), self.pointy+self.tolerance*(2.0/3.0)]
        create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get(),[0],2)
        self.im.set_array(image)
#        self.FigSubPlot.axis('off')
        self.canvas.show()
        time.sleep(1)
        self.scaler = 1
        Writer = animation.writers['ffmpeg']
        self.writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=-1)
        
        if self.zoom_flag == 1:
            self.init_x = 3.0
            self.init_y = self.init_x*(2.0/3.0)
    
            
            self.absxlim = self.init_x
            self.absylim = self.init_y
            s=((-self.tolerance/2.0)/-self.init_x)**(1.0/self.frame_num)
            sy = ((-self.ytolerance/2.0)/-self.init_y)**(1.0/self.frame_num)
        save_toggle = 1
        start = time.time()
        if save_toggle == 1:
            self.image_lib = {}
#            self.images = []
            group = 0
            self.image_lib['%s'%(group)] = []
            for it in range(0,self.frame_num):
                image = np.zeros((1000, 2000), dtype=np.uint8)
                if self.zoom_flag == 1:
                    self.absxlim = self.absxlim*s
                    self.absylim = self.absylim*sy
                    self.image_lib['%s'%(group)].append(create_fractal(self.pointx-self.absxlim, self.pointx+self.absxlim, self.pointy-self.absylim, self.pointy+self.absylim, image, self.iter_var.get(),[0],2))
#                    self.images.append(create_fractal(self.pointx-self.absxlim, self.pointx+self.absxlim, self.pointy-self.absylim, self.pointy+self.absylim, image, self.iter_var.get(),0))
                    if self.const_flag == 0:
                        self.iter = self.iter+(self.iter_target-self.iter_init)/(self.frame_num-1)
                        self.iter_var.set(self.iter)
                elif self.iter_flag == 1:
#                    self.images.append(create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get()),0)
                    self.image_lib['%s'%(group)].append(create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get(),[0],2))
                    self.iter = self.iter+(self.iter_target-self.iter_init)/(self.frame_num-1)
                    self.iter_var.set(self.iter)
                elif self.chan_flag == 1:
                    self.image_lib['%s'%(group)].append(create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get(),[0],2))
                if it/50.0 == float(int(it/50.0)) and it/50.0 != 0:
                    np.save('data%s.npy'%(group),self.image_lib['%s'%(group)])
                    print it,time.time()-start,self.iter_var.get()
                    group += 1
                    self.image_lib['%s'%(group)] = []
                   
#         self.images.append(create_fractal(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], image, self.iter_var.get(),-2.0+4.0*self.frame_num/(it+1)))
            if (it+1)/50.0 != float(int(it/50.0)):
                np.save('data%s.npy'%(group),self.image_lib['%s'%(group)])
                print "shortened clip"
            self.max_g = group
        else:
            self.image_lib = {}
            self.max_g = 0
            for data in glob.glob('*.npy'):
                group = map(int,data[4:-4])
                if self.max_g < group[0]:
                    self.max_g = group[0]
                self.image_lib['%s'%(group[0])] = np.load(data)
#        self.images = []
#        for g in range(0,self.max_g):
#            for it in range(0,len(self.image_lib['%s'%(group[0])])):
#                self.images.append(self.image_lib['%s'%(group[0])][it])           
#np.save('data.npy',self.images)
#            self.scaler = self.scaler*0.9
#            self.images = np.load('data.npy')
        print "Animation starts at", time.time()-start
        for self.g in range(0,self.max_g+1):
            self.frame_count = 0
            self.images = self.image_lib['%s'%(self.g)]
            self.ani = animation.FuncAnimation(self.Fig,self.update_zoom,repeat=False,frames=len(self.images))
            self.ani.save('test%s.mp4'%(self.g),writer=self.writer,dpi=120)
            print time.time()-start,"done",self.g
        print "done"
            

    
    def update_zoom(self,*args):
        if self.frame_count < len(self.images):
#            self.im.set_array(self.images[self.frame_count])
            self.im.set_array(self.images[self.frame_count])
            self.frame_count += 1
        else:
            pass
#        if self.frame_count/10.0 == float(int(self.frame_count/10.0)):
#            self.canvas.show()
#        return self.im

    def _on_mousewheel(self,event):
        if self._job != None:
            self.master.after_cancel(self._job)
            self._job = None
        inc = int((event.delta/120)*(1.0/2.0)*np.log(self.iter_var.get())**2)
        val = self.iter_var.get() + inc
        if val < 5 and np.sign(inc) == -1:
            self._job = None
            return
        elif val < 5 and np.sign(inc) == 1:
            self.iter_var.set(self.iter_var.get()+1)
        else:
            self.iter_var.set(val)
        self.menubar.entryconfigure(3,label=self.iter_var.get())
        self._job = self.master.after(400, self.graph_update)
        
    def pan_update(self,event):
#        print [self.FigSubPlot.get_xlim()[0],self.FigSubPlot.get_xlim()[1]],self.rxlim, [self.FigSubPlot.get_ylim()[0],self.FigSubPlot.get_ylim()[1]],self.rylim
        if [self.FigSubPlot.get_xlim()[0],self.FigSubPlot.get_xlim()[1]] != self.rxlim or [self.FigSubPlot.get_ylim()[0],self.FigSubPlot.get_ylim()[1]] != self.rylim:   

            self.graph_update()
        else:
            return
    
#    def slider_iter(self,*args):
#        if self.iter_var.get() == self.prev_iter:
#            self.graph_update()
#        else:
#            self.prev_iter = self.iter_var.get()
#            self.master.after(1000,self.slider_iter)

    def graph_update(self,*args):
        dimensions = (self.FigSubPlot.get_xlim()[1]-self.FigSubPlot.get_xlim()[0], self.FigSubPlot.get_ylim()[0]-self.FigSubPlot.get_ylim()[1])
        self.xlim = [(self.FigSubPlot.get_xlim()[0]*(self.xlim[1]-self.xlim[0])/(self.rxlim[1]-self.rxlim[0]))+self.xlim[0],(self.FigSubPlot.get_xlim()[1]*(self.xlim[1]-self.xlim[0])/(self.rxlim[1]-self.rxlim[0]))+self.xlim[0]]
#        print (self.rylim[0]-self.FigSubPlot.get_ylim()[0])
        self.ylim = [self.ylim[0]-((self.rylim[0]-self.FigSubPlot.get_ylim()[0])*(self.ylim[1]-self.ylim[0])/(self.rylim[1]-self.rylim[0])),self.ylim[0]-((self.rylim[0]-self.FigSubPlot.get_ylim()[1])*(self.ylim[1]-self.ylim[0])/(self.rylim[1]-self.rylim[0]))]
#        print self.ylim
        if dimensions[0]/dimensions[1] >= 2:
            #x bigger
            size = (int(dimensions[1]*2000/dimensions[0]),2000)
        else:
            size = (1000,int(dimensions[0]*1000/dimensions[1]))
        self.FigSubPlot.clear()
        image = np.zeros(size, dtype=np.uint8)
#
        create_fractal(self.xlim[0], self.xlim[1], -self.ylim[1], -self.ylim[0], image, self.iter_var.get(),[0],2)
#        self.im.set_array(image)
        self.FigSubPlot.axis('off')
        self.im = self.FigSubPlot.imshow(image,cmap=self.color_var.get())
        self.canvas.show()
        self.rxlim = [self.FigSubPlot.get_xlim()[0],self.FigSubPlot.get_xlim()[1]]
        self.rylim = [self.FigSubPlot.get_ylim()[0],self.FigSubPlot.get_ylim()[1]]
class NavigationToolbar(NavigationToolbar2TkAgg):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2TkAgg.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

root = Tk()
screen_resolution = (root.winfo_screenwidth(),0.98*root.winfo_screenheight())
root.geometry("%dx%d+%d+%d" % (screen_resolution[0], 
                               screen_resolution[1]-75,
                               -10,0))

app=App(root)
root.mainloop()
