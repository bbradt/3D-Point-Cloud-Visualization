# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from matplotlib import animation

DEF_ORIGIN = np.zeros((3,))
DEF_VERTICES = np.array([])

DEF_SIZE = 64
DEF_WORLD_SHAPE = (DEF_SIZE, DEF_SIZE, DEF_SIZE)
DEF_CANVAS_SHAPE = (DEF_SIZE, DEF_SIZE)

BASES = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
])



DEF_CUBE_WIDTH = 32
DEF_FOCAL_LEN = 1
DEF_TRANSLATION = np.array([0,0,0])

INTERPOLATE = np.ceil(np.sqrt(2*(DEF_FOCAL_LEN*DEF_CUBE_WIDTH)**2))

class Camera:
    def __init__(self, focal_len=DEF_FOCAL_LEN, translation=DEF_TRANSLATION):
        self.focal_len = focal_len
        self.projection = self.get_projection()
        pass
    def get_projection(self):
        return np.matrix([[self.focal_len, 0, 0, 0],
                          [0,self.focal_len,0,0],
                           [0,0,1,0]]
                         )

    def project(self, obj3d):
        unhomogenous_vertices = []
        for vertex in obj3d.vertices:
            homogenous = np.concatenate([vertex, np.ones((1,))])
            projected_vertex = np.asarray(self.projection.dot(homogenous)).flatten()
            unhomogenous_vertices.append(projected_vertex)
        division_factor = np.max([v[2] for v in unhomogenous_vertices])
        return [p[:2] for p in unhomogenous_vertices]

class Obj3d:
    def __init__(self, origin=DEF_ORIGIN, vertices=DEF_VERTICES):
        self.origin = origin
        self.vertices = vertices
    def rotate(self, alpha, beta, gamma, point=np.zeros((3,))):
        placeolder = [vertex-point for vertex in self.vertices]
        return [np.asarray(np.matrix([[1,0,0],
                          [0,np.cos(alpha),-np.sin(alpha)],
                          [0,np.sin(alpha),np.cos(alpha)]
                         ]).dot(
                np.matrix([[np.cos(beta),0,np.sin(beta)],
                           [0, 1, 0],
                           [-np.sin(beta),0,np.cos(beta)]
                        ])).dot(
                np.matrix([[np.cos(gamma),-np.sin(gamma),0],
                            [np.sin(gamma),np.cos(gamma),0],
                            [0,0,1]])).dot(vertex)).flatten() + point for vertex in placeolder]
                
        
        
        
class Cube3d(Obj3d):
    def __init__(self,origin=DEF_ORIGIN, w=DEF_CUBE_WIDTH):
        between = []
        bottomfrontleft = origin+np.array([0,0,0])
        # from front left to ront right
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(bottomfrontleft+np.array([1,0,0])*l)
        bottomfrontright = origin+np.array([w,0,0])
        # from front left to back left
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(bottomfrontleft+np.array([0,1,0])*l)
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(bottomfrontright+np.array([0,1,0])*l)
        bottombackleft = origin+np.array([0,w,0])
        # from front right to back right
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(bottombackleft+np.array([1,0,0])*l)
        bottombackright = origin+np.array([w,w,0])
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(bottomfrontleft+np.array([0,0,1])*l)
            between.append(bottomfrontright+np.array([0,0,1])*l)
            between.append(bottombackleft+np.array([0,0,1])*l)
            between.append(bottombackright+np.array([0,0,1])*l)
        topfrontleft = origin+np.array([0,0,w])
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(topfrontleft+np.array([1,0,0])*l)
        topfrontright = origin+np.array([w,0,w])
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(topfrontleft+np.array([0,1,0])*l)
        topbackleft = origin+np.array([0,w,w])
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(topfrontright+np.array([0,1,0])*l)
        for l in np.linspace(0,w,num=INTERPOLATE):
            between.append(topbackleft+np.array([1,0,0])*l)
        topbackright = origin+np.array([w,w,w])
        vertices = [
                bottomfrontleft,
                bottomfrontright,
                bottombackleft,
                bottombackright,
                topfrontleft,
                topfrontright,
                topbackleft,
                topbackright
        ] + between
        self.width = w
        Obj3d.__init__(self,origin=origin, vertices=vertices)

FRAMES = 100000
Xo =16
Yo =16
Zo =0
Rxo = Xo+DEF_CUBE_WIDTH/2
Ryo = Yo+DEF_CUBE_WIDTH/2
Rzo = Zo+DEF_CUBE_WIDTH/2
if __name__=='__main__':
    cube3d = Cube3d(origin=np.array([Xo,Yo,Zo]))
    camera=Camera()
    canvas = np.zeros(DEF_CANVAS_SHAPE)
    projected = camera.project(cube3d)
    for vertex in projected:
        x, y = vertex[0], vertex[1]
        if x <= canvas.shape[0] and y <= canvas.shape[1]:
            canvas[int(x)][int(y)] = 1
    myobj=plt.imshow(canvas,cmap='bone')
    def init():
        myobj.set_data(canvas)
    def animate(i):
        mylinspace = np.linspace(0,2*np.pi,FRAMES)
        r = mylinspace[i]
        cube3d.vertices = cube3d.rotate(r,r/3,r/4,np.array([Rxo,Ryo,Rzo]))
        projected = camera.project(cube3d)
        canvas = np.zeros(DEF_CANVAS_SHAPE)
        for vertex in projected:
            x, y = vertex
            if x <= canvas.shape[0] and y <= canvas.shape[1]:
                canvas[int(x)][int(y)] = 1
        myobj.set_data(canvas)
        plt.draw()
        plt.show()
    fig = plt.figure()
    anim= animation.FuncAnimation(fig, animate,frames=FRAMES,init_func=init, interval=1, repeat=True, blit=False)
    
