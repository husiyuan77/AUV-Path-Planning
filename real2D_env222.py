
"""
Reinforcement learning maze example.

"""
from dis import dis
import numpy as np
import numpy
import sys
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D
import math,logging
import csv,os
from scipy.interpolate import griddata
from itertools import groupby
from gym.spaces import Discrete,Box
from netCDF4 import Dataset
# START_POINT  = np.array([9.0,18.0])
# GOAL_POINT= np.array([36.0,30.0])
GOAL_POINT = np.array([1.0,1.0])
START_POINT= np.array([40.0,40.0])
# START_POINT = np.array([-7,0.0])
# GOAL_POINT = np.array([5,0])

MIN = 1
MAX = 40
record_dir = r'./history_point'
if not os.path.exists(record_dir):
    os.mkdir(record_dir)
class Env():
    def __init__(self):
        self.observation_space = Box(low=0.0, high=19.0, shape=(4,))
        self.action_space = Discrete(8)

        self.current_point = START_POINT
        self.current_pointwithflow = START_POINT.copy()
        self.goal = GOAL_POINT
        self.start = START_POINT
        self.start_togoal = GOAL_POINT-START_POINT
        self.step_counter = 0
        self.steplength= 1#lucheng per time
        # self.counter = 1
        self.speed = 1.2
        self.width = 3
        self.height = 3
        self.t = 0.4
        self.time = 0.5


        self.X, self.Y = np.meshgrid(np.arange(0, 43, 1), np.arange(0, 43, 1))
        self.U , self.V  = self.current_data()
        self.obs = self.obs_rectangle()
        self.circlelist = self.getcircles()
        self.r = 2
    @staticmethod
    def getcircles():
        obs_rectangle = [
            [3,8],
            [12,7],
            [26,25],
            [18,33],
            [26,6],
            [16,20],
            [25,15],
            [16,11],
            [32,32],
            [32,20],
            [35,5],
            [5,35],
            [10,29],
            [7,20],
            [19,25],
            
            [30,5], #0912 19:19:37
            [6.5,25],
            [20.5,30],
            [10,15],
            [23,20],
        ]
        # for i in range(15):
        #
        #     random_x = random.random()*30+4
        #     random_y = random.random() * 30+4
        #     obs_rectangle.append([random_x,random_y])
        return obs_rectangle
    def iscollision(self):
        x = self.current_point[0]
        y = self.current_point[1]
        flag = False
        #print(self.obs[0])
        # for i in range(len(self.obs[0])):
        #     dis = (x-self.obs[0][i])**2+(y-self.obs[1][i])**2
        #     if dis<0.5:
        #         flag = True
        #         break
        for pos in self.circlelist:
            dis1 = np.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            # dis2 = np.sqrt((pos[0] - GOAL_POINT[0]) ** 2 + (pos[1] - GOAL_POINT[1]) ** 2)
            if dis1 < self.r:
                return True
        return flag

    def obs_rectangle(self):
        obstacle_x = []
        obstacle_y = []
        obslist=[]
        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                aa = str(type(self.U[i][j]))
                if aa != '<class \'numpy.float32\'>':
                    obstacle_x.append(i)
                    obstacle_y.append(j)
                    self.U[i][j]= 0
                    self.V[i][j] = 0
        # obslist.append(obstacle_x)
        # obslist.append(obstacle_y)
#         x = np.linspace(0, self.U.shape[0], 1)
#         y = np.linspace(0, self.U.shape[1], 1)
#         x, y = np.meshgrid(x, y)  # 20*20的网格数据
#         # newx = np.linspace(0, self.U.shape[0], 0.5)
#         # newy = np.linspace(0, self.U.shape[1], 0,5)
#         newfunc = interpolate.interp2d( x,  y, self.U , kind='cubic')
# # 计算 100*100 网格上插值
#         xnew = np.linspace(-1,1,100)
#         ynew = np.linspace(-1,1,100)
#         fnew = newfunc(xnew ,ynew)
        
        #print(obslist)
        return obslist



    @staticmethod
    def current_data():
        nc_obj = Dataset('/home/hanj/hjm_project/chinasea_201801_c.nc')
        uu = np.transpose(nc_obj.variables['u'][1, :, :])
        vv = np.transpose(nc_obj.variables['v'][1, :, :])
        uu = uu[60:101, 40:81]
        vv = vv[60:101, 40:81]
        return uu, vv

    def searchflow(self, k=0.09):
        x = self.current_point[0]
        y = self.current_point[1]

        x1= math.floor(x)
        y1 = math.floor(y)
        if x>=MAX  or y>=MAX:
            u =  np.array([self.U[x1][y1],self.V[x1][y1]])
            # v =  np.array(self.V[x1][y1])
            # uv = np.concatenate((u,v))
            return u
        x2 = math.floor(x)+1
        y2 = math.floor(y) + 1
        point_grid = self.current_point.copy()
        points = np.array([[x1,y1],[x1,y2],[x2,y1],[x2,y2]])

        values_u = np.array([self.U[x1][y1],self.U[x1][y2],self.U[x2][y1],self.U[x2][y2]])
        values_v = np.array([self.V[x1][y1], self.V[x1][y2], self.V[x2][y1], self.V[x2][y2]])
        u = griddata(points,values_u,point_grid,method='linear')
        v = griddata(points, values_v, point_grid, method='linear')

        uv = np.concatenate((u,v))
        return uv
    def reset(self):
        # self.counter = 1
        # self.history = {}
        self.step_counter = 0
        self.current_point = START_POINT.copy()
        local_area = self.pos_toimg().flatten()
        self.current_pointwithflow = np.concatenate((self.current_point,self.searchflow()))
        state = np.array(self.current_pointwithflow)
        return state
    def changestart(self):
        # self.counter = 1
        # self.history = {}
        self.step_counter = 0
        self.current_point = np.array([8.0,4.0])
        self.goal = GOAL_POINT
        self.start = self.current_point.copy()
        self.start_togoal = GOAL_POINT - self.start
        local_area = self.pos_toimg().flatten()
        self.current_pointwithflow = np.concatenate((self.current_point,self.searchflow(),local_area))
        state = np.array(self.current_pointwithflow)
        return state

    # def check_obstacle(self,x,y):
    #     flag = False
    #     for obs in self.obs_rectangle:
    #         if obs[0] <= x <= obs[0] + obs[2] \
    #                 and obs[1] <= y <= obs[1] + obs[3]:
    #             flag = True
    #             break
    #
    #     return flag


    def pos_toimg(self):
        img = np.ones((self.width,self.height))
        #pos = self.current_point.copy()
        centerx_index = 1
        centery_index = 1
        #if_obstacle = self.check_obstacle()
        for i in range(self.width):
            for j in range(self.height):
                if_obstacle = self.iscollision()#(self.current_point[0]-(centerx_index-i)*self.steplength ,self.current_point[1]-(centery_index-j)*self.steplength )
                # if (i==centery_index or j==centery_index) \
                #         else self.check_obstacle(self.current_point[0]-(centerx_index-i)*self.steplength/np.sqrt(2)/2,self.current_point[1]-(centery_index-j)*self.steplength/np.sqrt(2)/2)

                if if_obstacle:
                      #obstacle_reward
                    img[i][j]=0
                elif (self.current_point[0]-(centerx_index-i)*self.steplength) > MAX or (self.current_point[0]-(centerx_index-i)*self.steplength )<MIN or\
                        (self.current_point[1]-(centery_index-j)*self.steplength) > MAX or (self.current_point[1]-(centery_index-j)*self.steplength )<MIN:
                    img[i][j] =0
                else:
                    img[i][j]= 1
        #img[centerx_index][centery_index] = 1
        return img
    def step(self, action):

        action = self.is_action_valid(action)
        old_flow = self.searchflow()
        old_pos,position,act_spd = self.caculate_position(action,old_flow)
        position, ifvalid = self.is_position_valid(position)

        self.current_point = position
        #state = np.array(self.current_point)
        local_area = self.pos_toimg().flatten()
        self.current_pointwithflow = np.concatenate((self.current_point, self.searchflow()))
        state = np.array(self.current_pointwithflow)
        done = self.is_done(ifvalid)
        reward = self.caculate_reward(old_pos,done,old_flow,local_area,act_spd )

        self.step_counter = self.step_counter + 1


        #assert self.observation_space.contains(state), "%r (%s) invalid" % (state, type(state))
        return state, reward, done

    def is_action_valid(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        return action

     

    def caculate_position(self,action,old_flow):
        '''
        add ocean current information
        '''

        position = self.current_point.copy()
        old_pos = position.copy()
        #print(old_pos)
        basespeed = np.array([0.0,0.0])
        if action == 0: # up
            basespeed[0] = basespeed[0] + self.speed
        elif action == 1: # down
            basespeed[1] = basespeed[1] - self.speed
        elif action == 2: # left
            basespeed[0] = basespeed[0] - self.speed
        elif action == 3: # right
            basespeed[1] = basespeed[1] + self.speed
        elif action == 4: # down
            basespeed[0] = basespeed[0] - self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] - self.speed/np.sqrt(2)
        elif action == 5: # left
            basespeed[0] = basespeed[0] - self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] + self.speed/np.sqrt(2)

        elif action == 6: # forward
            basespeed[0] = basespeed[0] + self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] - self.speed/np.sqrt(2)
        else:  # forward
            basespeed[0] = basespeed[0] + self.speed/np.sqrt(2)
            basespeed[1] = basespeed[1] + self.speed/np.sqrt(2)
        position = position +self.time* (basespeed+old_flow)
        actual_spd = basespeed+old_flow
        #print(flow)
        #print(position)

        return old_pos,position,actual_spd

    def is_position_valid(self,position):
        valid = True
        for i in range(2):
            if position[i] > MAX:
                position[i] = MAX
                valid = False
            if position[i] < MIN:
                position[i] = MIN
                valid = False
        correct_position = position
        return correct_position,valid

    def get_actualtime(self,real_act,old_flow,getleast=False):
        a = np.linalg.norm(old_flow)
        #print(a)
        b = self.speed
        theta = 0 if getleast else self.get_cosvalue(old_flow,real_act)
        speed = np.sqrt(a**2+ b**2- 2*(a*np.sin(theta))**2+\
                 (1 if theta<np.pi/2 else -1)*2*a*np.sqrt((b**2-(a*np.sin(theta))**2)*np.cos(theta)**2))
        #print(np.linalg.norm(real_act))
        #time = np.linalg.norm(real_act)/cc
        return speed

    def get_cosvalue(self,old_flow,real_act):
        if np.dot(old_flow,real_act) != 0:
            cos_value = np.dot(old_flow, real_act) / (np.linalg.norm(real_act) * np.linalg.norm(old_flow))
            cos_value = round(cos_value,10)
            np.clip(cos_value,-1,1)
            #print(real_act, old_flow,cos_value)
            theta = math.acos(cos_value)
        else:
            theta = np.pi/2

        return theta
    def getrelative(self, obs):
        kk = obs.copy()
        slice1 = kk[:2]
        slice2 = kk[2:]
        rela_obs = np.concatenate((-GOAL_POINT[0:1]+slice1[0:1],-GOAL_POINT[1:2]+slice1[1:2],slice2*5))
        #print(rela_obs)
        return rela_obs
    def caculate_reward(self,old_pos,done,old_flow,local_sense,act_spd):

        real_action = self.current_point - old_pos
        goal_dic = GOAL_POINT - old_pos
        new_goaldic = GOAL_POINT - self.current_point
        Static_goaldic = GOAL_POINT - old_pos
        real_action = self.current_point - old_pos
        pos_contribution = real_action / self.start_togoal * np.abs(self.start_togoal)

        if done:
            reward = 100#+200/(self.step_counter)
        elif all(self.current_point == old_pos):
            reward = -10
        else:
            #time_reward = 1/(self.get_actualtime(real_action,old_flow)) - 0.9/((self.get_actualtime(real_action,old_flow,True)) )
            actual_speed = np.linalg.norm(act_spd)
            #norm_action = real_action/self.steplength
            old_dis =np.sqrt(np.sum((old_pos - GOAL_POINT) ** 2))
            new_dis = np.sqrt(np.sum((self.current_point - GOAL_POINT) ** 2))
            
                
            distance_reward = (old_dis-new_dis-self.time*self.speed )*10
            distance_reward = np.arctan(distance_reward)
            flow_reward = np.arctan(10 * actual_speed/self.speed -10)

            lamda1 = 2#1
            lamda2 = 1 #1.5
            # distance_reward = 1.5 * distance_reward if distance_reward < 0 else distance_reward
            # flow_reward = 1.5 * flow_reward if flow_reward < 0 else flow_reward
            reward = lamda1 * distance_reward + lamda2 * flow_reward
            # if new_dis<10:
            #     reward += 3
            # if new_dis<10:
            #     reward += 7
            reward = reward/(0.05*self.step_counter+1) if reward>0 and self.step_counter>50 else reward
            #print(actual_speed,distance_reward, flow_reward, reward)
 
        if self.iscollision()  :#local_sense[60]<0:  #move to obstacle
            reward = reward-20
            #print("obs")
        # elif np.min(local_sense)<0 and reward>0:
        #     reward = reward+10

        #print(distance_reward,flow_reward,reward)
        return reward

    def is_done(self,valid):
        # if all(self.current_point == GOAL_POINT):
        if np.sqrt(np.sum((self.current_point - GOAL_POINT) ** 2)) < 0.5:
            print("goal")
            return True
        # elif (not valid):
        #     print("out of boundary")
        #     return True
        else:
            return False
    def get_actualtime(self,real_act,old_flow,getleast=False):
        a = np.linalg.norm(old_flow)
        #print(a)
        b = self.speed
        theta = 0 if getleast else self.get_cosvalue(old_flow,real_act)
        speed = np.sqrt(a**2+ b**2- 2*(a*np.sin(theta))**2+\
                 (1 if theta<np.pi/2 else -1)*2*a*np.sqrt((b**2-(a*np.sin(theta))**2)*np.cos(theta)**2))
        #print(np.linalg.norm(real_act))
        #time = np.linalg.norm(real_act)/cc
        return speed
    def finalflow(self, X, Y, k=1.44):

        x = int(X)#self.current_point[0]
        y = int(Y)#self.current_point[1]
        x1= math.floor(x)
        y1 = math.floor(y)
        x2 = math.floor(x)+1
        y2 = math.floor(y) + 1
        point_grid = self.current_point.copy()
        points = np.array([[x1,y1],[x1,y2],[x2,y1],[x2,y2]])
        values_u = np.array([self.U[x1][y1],self.U[x1][y2],self.U[x2][y1],self.U[x2][y2]])
        values_v = np.array([self.V[x1][y1], self.V[x1][y2], self.V[x2][y1], self.V[x2][y2]])
        u = griddata(points,values_u,point_grid,method='linear')
        v = griddata(points, values_v, point_grid, method='linear')
        u = self.U[x1][y1]
        v = self.V[x1][y1]
        uv = np.concatenate((u, v))
        #print(u,v)
        return u,v

    def final(self,nresult_list):
        result = [list(g) for k, g in groupby(nresult_list, lambda x: x == '') if not k]
        kkk = result[0]
        m, n = np.zeros(len(kkk)), np.zeros(len(kkk))
        for i in range(len(kkk)):
            # print(nresult_list[1][0])
            m[i] = kkk[i][0]
            n[i] = kkk[i][1]
        # print(result[0])
        # plt.plot(m, n)
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        fig, ax = plt.subplots()
        x, y = np.meshgrid(np.arange(0, self.U.shape[0], 1),
                           np.arange(0, self.U.shape[1], 1)
                           )
        #plt.scatter(self.obs[0], self.obs[1], s=50, c='b', marker='s')
        # X, Y ,Z = np.meshgrid(np.arange(-20, 20, 8), np.arange(-20, 20, 4),np.arange(-15, 17, 8))
        ax.quiver(x, y, np.transpose(self.U), np.transpose(self.V),
                  width=0.001)  # ,headwidth= 1.5,headlength= 1,scale=40,units= 'inches')

        distlist = []
        for i in range(len(m) - 1):
            action = np.array([m[i + 1] - m[i], n[i + 1] - n[i]])
            dist = np.linalg.norm(action)*5500

            distlist.append(dist)

        # ax.quiverkey(q, X=5, Y=0.5, U=10,
        #              label='Quiver key, length = 10', labelpos='N')

        # ax.quiver(X, Y, z, 0.5 * self.u, 0.5 * self.v, 0.5 * self.w, length=1)
        # start = self.current_point
        goal = self.goal
        xx = np.array([START_POINT[0], GOAL_POINT[0]])
        yy = np.array([START_POINT[1], GOAL_POINT[1]])
        plt.plot(m, n)
        for rectangle_para in self.circlelist:
            ax.add_patch(
                plt.Circle(rectangle_para,self.r))
        #
        # plt.scatter(m, n, s=4, edgecolors='k')
        plt.xlabel('x(*0.5°)')
        plt.ylabel('y(*0.5°)')
        # plt.plot(xx, yy, '--', color='b')
        # print(START_POINT, GOAL_POINT)
        # ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
        # ax.plot(self.obstacle1[0:1], self.obstacle1[1:2], 'bo', markersize=7, markeredgecolor='k')
        # ax.plot(self.obstacle2[0:1], self.obstacle2[1:2], 'bo', markersize=7, markeredgecolor='k')
        ax.plot(self.start[0:1], self.start[1:2], 'go', markersize=10, markeredgecolor='k')
        ax.plot(goal[0:1], goal[1:2], 'r*', markersize=10, markeredgecolor='k')
        # ax.plot(goal[0:1], goal[1:2], 'r*', markersize=7, markeredgecolor='k')
        print(m.tolist())
        print(n.tolist())
        print(sum(distlist))
        print(len(distlist)*5500*0.5)

        # plt.grid()
        # ax.plot(11.5, 1.0, 'ro', markersize=7, markeredgecolor='k')
        plt.savefig('path' + '.png')
        
    def visualization(self):
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10,10, 1))
        q = ax.quiver(X, Y, self.u, self.v)

        for rectangle_para in self.obs_rectangle:
            ax.add_patch(
                plt.Rectangle(
                    (rectangle_para[0], rectangle_para[1]),
                    rectangle_para[2],
                    rectangle_para[3],
                    color='black',
                    alpha=1
                )
            )
        ax.plot(self.start[0:1], self.start[1:2], 'bo', markersize=12, markeredgecolor='k')
        goal = self.goal
        ax.plot(goal[0:1], goal[1:2], 'ro', markersize=7, markeredgecolor='k')
        return fig,ax
    def refresh(self,fig,ax):
        #fig, ax = plt.subplots()
        ax2 = ax.twinx()
        lines = ax2.add_patch(
            plt.Rectangle(
                (self.current_point[0]-5*self.steplength/6-self.steplength,self.current_point[1]-5*self.steplength/6-self.steplength),
                self.width*self.steplength/3,
                self.height*self.steplength/3,
                color='black',
                alpha =0.2
            )
        )
        # print(self.current_point,self.current_point[0] - 5 * self.steplength / 6 - self.steplength, self.current_point[
        #     1] - 5 * self.steplength / 6 - self.steplength)


        ax2.plot(self.current_point[0:1], self.current_point[1:2], 'go', markersize=7, markeredgecolor='k')
        #ax.plot(self.start[0:1], self.start[1:2], 'go', markersize=7, markeredgecolor='k')
        #ax.plot(self.obstacle1[0:1], self.obstacle1[1:2], 'bo', markersize=7, markeredgecolor='k')
        #ax.plot(GOAL_POINT[0:1], GOAL_POINT[1:2], GOAL_POINT[2:], 'r*', markersize=7, markeredgecolor='k')
        plt.axis([-10,10,-10,10])
        #plt.show()
        plt.axis('off')
        plt.pause(0.01)
        lines.remove()
        return ax2

    def clearax(self):
        plt.close()
        fig,ax = plt.subplots()
        X, Y = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
        q = ax.quiver(X, Y, self.u, self.v)

        for rectangle_para in self.obs_rectangle:
            ax.add_patch(
                plt.Rectangle(
                    (rectangle_para[0], rectangle_para[1]),
                    rectangle_para[2],
                    rectangle_para[3],
                    color='black',
                    alpha=1
                )
            )
        ax.plot(self.start[0:1], self.start[1:2], 'bo', markersize=12, markeredgecolor='k')
        goal = self.goal
        ax.plot(goal[0:1], goal[1:2], 'ro', markersize=7, markeredgecolor='k')
        return ax



