import vizdoom as vzd
#Previamente "from vizdoom import *" cambio por recomendacion del seba, para tener autocompletar.
import os
import random
import time

game = vzd.DoomGame()

#game.load_config("githubVizDoom/ViZDoom/scenarios/basic.cfg")  
game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg")) 
#Ambas formas de cargar config funcionan, la 1ra es la que aparece en la pag de vizdoom, la segunda es una recomendacion del seba.

game.init()

shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        print ("\treward:", reward)
        time.sleep(0.02)
    print ("Result:", game.get_total_reward())
    time.sleep(2)