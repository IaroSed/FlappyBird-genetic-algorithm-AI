# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:15:56 2019

@author: iasedric
"""


# Keras is to be imported only once in the beggining.

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils.vis_utils import plot_model
from keras import backend as K


import pygame
import random
import pandas as pd
import heapq
import numpy as np
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Defining constants
NUMBER_MODELS = 20

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 480
TEXT_ZONE = 100

SINK_SPEED = 10
CLIMB_SPEED = 30
   
PIPES_GAP = 100
PIPES_WIDTH  = 80
PIPES_SPEED = 5

PROBA = 0.85
DELTA = 0.5

# Loading sprites
pygame.display.set_caption("Flappy Bird")
bg_sprite = pygame.image.load('bg.jpg')


# Creating classifiers.
## TEMPORARY. Move to Bird class.

classifier = [0] * NUMBER_MODELS

for i in range(0,NUMBER_MODELS):
    # Initialising the ANN
    classifier[i] = Sequential()
    # Adding the input layer and the first hidden layer
    classifier[i].add(Dense(output_dim = 3, kernel_initializer='random_uniform', activation = 'relu', input_dim = 3))
    # Adding the output layer
    classifier[i].add(Dense(output_dim = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
    


# Creating a holder to remember the 10 best models
best_classifiers = [0] * 10

for i in range(0,10):
    # Initialising the ANN
    best_classifiers[i] = Sequential()
    # Adding the input layer and the first hidden layer
    best_classifiers[i].add(Dense(output_dim = 3, kernel_initializer='random_uniform', activation = 'relu', input_dim = 3))
    # Adding the output layer
    best_classifiers[i].add(Dense(output_dim = 1, kernel_initializer='random_uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    best_classifiers[i].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 




# Defining classes
class Bird():
    
    def __init__(self):
        self.x = SCREEN_WIDTH/3
        self.y = SCREEN_HEIGHT/2
        self.width = 32
        self.height = 32
        self.alive = True       
        self.isClimbing = False
        self.fitness = 0
        self.score = 0
        
        self.body = pygame.Rect(self.x,self.y,self.width,self.height)
        self.bird_sprite = [pygame.image.load('bird_wing_up.png'), pygame.image.load('bird_wing_down.png')]

        
    def flapswings(self):
        #self.start_climb = pygame.time.get_ticks()
        self.isClimbing = True
        
        if  self.isClimbing and self.alive:
            #self.time_in_climb = pygame.time.get_ticks() - self.start_climb
            self.y -= CLIMB_SPEED
            self.body = pygame.Rect(self.x,self.y,self.width,self.height)

            
    def sinks(self):
        
        if self.alive:
            self.y += SINK_SPEED
            self.body = pygame.Rect(self.x,self.y,self.width,self.height)
            self.isClimbing = False
        
    def draw(self):
        if self.alive:
            if self.isClimbing:
                i = 0
            else:
                i = 1
            self.fitness += 5
                       
            win.blit(self.bird_sprite[i], (self.x,self.y))

        

class Pipes():
    
    def __init__(self, x = SCREEN_WIDTH + PIPES_WIDTH):
        
        self.x = x
        self.height_top = random.randint(50,300)
        self.y_top = 0
        self.y_bottom = self.height_top + PIPES_GAP
        
        self.pipe_top = pygame.Rect(self.x,self.y_top,PIPES_WIDTH,self.height_top)
        self.pipe_bottom = pygame.Rect(self.x,self.y_bottom,PIPES_WIDTH,SCREEN_HEIGHT - PIPES_GAP - self.height_top)

        
    def move(self):
        
        if self.x > -PIPES_WIDTH:
            self.x -= PIPES_SPEED
            self.pipe_top = pygame.Rect(self.x,self.y_top,PIPES_WIDTH,self.height_top)
            self.pipe_bottom = pygame.Rect(self.x,self.y_bottom,PIPES_WIDTH,SCREEN_HEIGHT - PIPES_GAP - self.height_top)

        else:
            self.x = SCREEN_WIDTH + PIPES_WIDTH
            self.height_top = random.randint(50,300)
            self.y_bottom = self.height_top + PIPES_GAP
            self.pipe_top = pygame.Rect(self.x,self.y_top,PIPES_WIDTH,self.height_top)
            self.pipe_bottom = pygame.Rect(self.x,self.y_bottom,PIPES_WIDTH,SCREEN_HEIGHT - PIPES_GAP - self.height_top)
            

        
    def draw(self):
        # Draw top pipe
        pygame.draw.rect(win, (34, 139, 34), self.pipe_top)
        # Draw bottom pipe
        pygame.draw.rect(win, (34, 139, 34), self.pipe_bottom)



# Defining functions
def crossover(fitness):
    

    best_fitness_temp = [0] * 30
    print("1: " + str(best_classifiers[0].get_weights()[0][0]))

    best_fitness_temp = best_fitness + fitness

        
    best_parents_int = []
    best_parents_int = heapq.nlargest(30, enumerate(best_fitness_temp), key=lambda x: x[1])
    
    print("The last best is: " + str(heapq.nlargest(1, enumerate(best_fitness_temp), key=lambda x: x[1])[0][1]) + " the best is: " + str(best_fitness[0]))
    
    # Updating the best classifiers
    for i in range(0,10):
        best_classifiers[i].set_weights((best_classifiers+classifier)[best_parents_int[i][0]].get_weights())
        best_fitness[i] = best_fitness_temp[best_parents_int[i][0]]


    #Replacing the first 5 classifiers by the best and no mutation 
    for i in range(0,5):
        classifier[i].set_weights(best_classifiers[i].get_weights())

    # Keeping some bad ones for diversity   
    for i in range(5,10):
        classifier[i].set_weights((best_classifiers+classifier)[best_parents_int[i+5][0]].get_weights())

    #Replacing the first 5 classifiers by the best and mutation authorized
    for i in range(10,15):
        classifier[i].set_weights(best_classifiers[i-10].get_weights())
        
    # Creating some cross parents
    CPw = [0] * 5
    for i in range(15,20):
        classifier[i].set_weights(best_classifiers[i-15].get_weights()) 
        CPw[i-15] = best_classifiers[i-15].get_weights()[2]
 
    for i in range(15,20):
        classifier[i].set_weights([best_classifiers[i-15].get_weights()[0], best_classifiers[i-15].get_weights()[1], CPw[14-i] , best_classifiers[i-15].get_weights()[3]])  
        
    print("6: " + str(best_classifiers[0].get_weights()[0][0]))

        
def mutate():
    
    
    #Introducing mutations: with a probability of (100% - PROBA) each weight can be changed by a number between -DELTA to +DELTA.
    for i in range(5,NUMBER_MODELS):
        
        A = classifier[i].get_weights()[0][0]
        B = classifier[i].get_weights()[0][1]
        C = classifier[i].get_weights()[0][2]
        D = classifier[i].get_weights()[2]
    
        #number_changes = 0
        
        for j in range(0,3):
            
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                A[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C)]) , np.zeros(3, dtype=float), D , np.zeros(1, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                B[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C)]) , np.zeros(3, dtype=float), D , np.zeros(1, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                C[j] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C)]) , np.zeros(3, dtype=float), D , np.zeros(1, dtype=float)])
                #number_changes += 1
            if random.uniform(0,1) > PROBA:
                change = random.uniform(-DELTA,DELTA)
                D[j][0] += change
                classifier[i].set_weights([np.array([list(A),list(B),list(C)]) , np.zeros(3, dtype=float), D , np.zeros(1, dtype=float)])
                #number_changes += 1
    print("7: " + str(best_classifiers[0].get_weights()[0][0]))
 
#### End of functions and classes   




  
    
### Main    
    
pygame.init()
pygame.font.init()



#Opening the screen
win = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT+TEXT_ZONE))        
        
b = list(range(0,NUMBER_MODELS))
X = list(range(0,NUMBER_MODELS))
flap = list(range(0,NUMBER_MODELS)) 
        
for i in range(0,NUMBER_MODELS):
    b[i] = Bird()
    b[i].y += 4*i
    X[i] = pd.DataFrame([[0 , 0, 0]], columns=['DistanceGap','HeightGap', 'HeightBird'])
    
p1 = Pipes(SCREEN_WIDTH + PIPES_WIDTH)
p2 = Pipes(1.5*SCREEN_WIDTH + PIPES_WIDTH)

alive = []
fitness = []

Score = 0
Generation = 1

myfont = pygame.font.SysFont('Comic Sans MS', 20)

    
best_fitness = [0] * 10
the_best_fitness = 0

run = True

while run:
    pygame.time.delay(10)
    
    generation_info = myfont.render('Generation ' + str(Generation) + " Best fitness: " + str(best_fitness[0]) , False, (255, 255, 255))
    
    win.fill((0,0,0))
    win.blit(bg_sprite, (0,0))
    
    
    p1.draw()
    p2.draw()
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            run = False
        
    for i in range(0,NUMBER_MODELS):
        
        
        b[i].draw()        
        # This
        
        if (p1.x > p2.x and p2.x + PIPES_WIDTH + 1.5*b[i].width > b[i].x):
            X[i].iloc[0,0] = p2.x + PIPES_WIDTH - b[i].x + 1.5*b[i].width
            X[i].iloc[0,1] = p2.height_top + PIPES_GAP/2
        elif (p1.x > p2.x and p2.x + PIPES_WIDTH + 1.5*b[i].width < b[i].x):
            X[i].iloc[0,0] = p1.x +  PIPES_WIDTH  - b[i].x + 1.5*b[i].width
            X[i].iloc[0,1] = p1.height_top + PIPES_GAP/2
        elif (p2.x > p1.x and p1.x + PIPES_WIDTH + 1.5*b[i].width > b[i].x):
            X[i].iloc[0,0] = p1.x + PIPES_WIDTH  - b[i].x + 1.5*b[i].width
            X[i].iloc[0,1] = p1.height_top + PIPES_GAP/2
        elif (p2.x > p1.x and p1.x + PIPES_WIDTH + 1.5*b[i].width < b[i].x):
            X[i].iloc[0,0] = p2.x + + PIPES_WIDTH  - b[i].x + 1.5*b[i].width
            X[i].iloc[0,1] = p2.height_top + PIPES_GAP/2
            
        X[i].iloc[0,2] = b[i].y + b[i].height/2
        
        #Draw DistanceGap for each bird
        #pygame.draw.line(win, (255,0,0), (b[i].x+b[i].width, X[i].iloc[0,1]), (X[i].iloc[0,0] + b[i].x, X[i].iloc[0,1]))
        
        #Draw end of distance between pipes at which birds are aiming
        pygame.draw.circle(win, (255,0,0), (int(X[i].iloc[0,0] + b[i].x), int(X[i].iloc[0,1])), 2)

        
        #Draw HeightBird for each bird
        #pygame.draw.line(win, (255,0,0), (b[i].x+b[i].width, 0), (b[i].x+b[i].width, X[i].iloc[0,2]))



        # with a Sequential model
        #get_0_layer_output = K.function([best_classifiers[0].layers[0].input],[best_classifiers[0].layers[0].output])
        #layer0_output = get_0_layer_output([ X[i].iloc[0,0],  X[i].iloc[0,1],  X[i].iloc[0,2]])[0]
        
        #print(layer0_output)


        # Predicting the Test set results
        flap[i] = classifier[i].predict(X[i])
        #Exit_SB.write(str(i) + "^" + str(b[i].x) + "^" + str(b[i].y) + "^" + str(X[i].iloc[0,0]) + "^" + str(X[i].iloc[0,1]) + "^" + str(flap[i]) + "^" + str((flap[i] > 0.5)) )
        #Exit_SB.write("\n")
		

        flap[i] = (flap[i] > 0.5)
        # To this should be in a separate method
    
    
        #if keys[pygame.K_SPACE]:
        if flap[i]:
        
            #print("Flapping...")
            b[i].flapswings()
        
        else:
            #print("Sinking...")
            b[i].sinks()
        
        
        if (b[i].body.colliderect(p1.pipe_top) == True or b[i].body.colliderect(p1.pipe_bottom) == True) \
        or (b[i].body.colliderect(p2.pipe_top) == True or b[i].body.colliderect(p2.pipe_bottom) == True) \
        or (b[i].y < 0 or b[i].y + b[i].height> SCREEN_HEIGHT): 
            b[i].alive = False
        
        if ((b[i].x == p1.x + PIPES_WIDTH or b[i].x == p2.x + PIPES_WIDTH) and run == True):
            b[i].score +=1
            #print(Score)
            
        #Testing if the bird is still alive
        alive.append(b[i].alive)
        fitness.append(b[i].fitness)

    #print("Number alive: " + str(sum(alive)))    
    
    p1.move()
    p2.move()
        
    if (sum(alive) == 1):
        #Showing fitness of the last alive
        fitness_info = myfont.render('Fitness of the last alive: ' + str(b[[i for i, x in enumerate(alive) if x][0]].fitness) , False, (255, 255, 255))
        win.blit(fitness_info,(5,SCREEN_HEIGHT + 25))
        #Score
        score_info = myfont.render('Score: ' + str(b[[i for i, x in enumerate(alive) if x][0]].score) , False, (0, 0, 0))
        win.blit(score_info,(5, 5))
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
            
    # If all the birds are dead        
    if (not any(alive)):
        
        Score = 0
        
        # Crossover: the best two parents exchange their genes (weights)
        crossover(fitness)
        
        #Introducing mutations: with a probability of 5% each weight can be changed by a number between -0.1 to +0.1
        mutate()
        
        Generation += 1
        #print("Showing new Generation " + str(Generation) + " Best fitness: " + str(best_fitness[0]))
        
        for i in range(0,NUMBER_MODELS):
            b[i].y = SCREEN_HEIGHT/2 + 4*i
            b[i].fitness = 0
            b[i].score = 0
            b[i].alive = True

            
        p1.height_top = random.randint(50,300)
        p2.height_top = random.randint(50,300)
        p1.y_bottom = p1.height_top + PIPES_GAP 
        p2.y_bottom = p2.height_top + PIPES_GAP 
        p1.x = SCREEN_WIDTH - PIPES_WIDTH
        p2.x = 1.5*SCREEN_WIDTH - PIPES_WIDTH
        
    alive = []
    fitness = []
    
    #Showing generation
    win.blit(generation_info,(5,SCREEN_HEIGHT + 2))
    
   
    
    
    
    pygame.display.update()
    

pygame.quit()


