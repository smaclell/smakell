import json
import os
import random
import bottle
from gym import spaces
import numpy as np
import math

from .api import ping_response, start_response, move_response, end_response
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

# self.action_space = spaces.Discrete(4)
# self.observation_space = spaces.Box(low=0,high=255, shape=(LAYER_WIDTH, LAYER_HEIGHT, NUM_LAYERS), dtype=np.uint8)

@bottle.route('/')
def index():
    return '''
    Battlesnake documentation can be found at
       <a href="https://docs.battlesnake.io">https://docs.battlesnake.io</a>.
    '''

@bottle.route('/static/<path:path>')
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.

    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root='static/')

@bottle.post('/ping')
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()

@bottle.post('/start')
def start():
    data = bottle.request.json

    """
    TODO: If you intend to have a stateful snake AI,
            initialize your snake state here using the
            request's data if necessary.
    """
    print(json.dumps(data))

    color = "#56EEF4"
    headType = 'silly'
    tailType = 'sharp'

    return start_response(color, headType, tailType)


BOARD_WIDTH = 11
BOARD_HEIGHT = 11


NUM_LAYERS = 6
LAYER_WIDTH = 39
LAYER_HEIGHT = 39

model = PPO2.load('/snake/model.pkl')

def prepareObservations(you, snakes, food, orientation):
  head = you['body'][0]
  hx = head['x']
  hy = head['y']
  yourLength = len(you['body'])

  observations = []
  def assign(point, layer, value):
      x = point['x']
      y = point['y']
      x = (x - hx) * (-1 if orientation & 1 != 0 else 1)
      y = (x - hy) * (-1 if orientation & 2 != 0 else 1)
      x += LAYER_WIDTH / 2
      y += LAYER_HEIGHT / 2
      if x > 0 and x < LAYER_WIDTH and y > 0 and y < LAYER_HEIGHT:
          observations[ math.floor(x*(LAYER_HEIGHT*NUM_LAYERS) + y*NUM_LAYERS + layer)] = value

  for snake in snakes:
      body = snake['body']
      assign(body[0], 0, snake['health'])
      i = 0
      for part in body:
          i += 1
          assign(part, 1, 1)
          assign(part, 2, min(i, 255))

      if snake['id'] != you['id']:
          assign(body[0], 3, 1 if len(body) >= yourLength else 0)

  for pellet in food:
      assign(pellet, 4, 1)

  for x in range(0, BOARD_WIDTH):
      for y in range(0, BOARD_WIDTH):
          assign({ 'x': x, 'y': y }, 5, 1)

  return observations


up = 'up'
down = 'down'
left = 'left'
right = 'right'
moves = [up, down, left, right]

def getDirection(index, orientation):
  action = moves[index]
  if (orientation & 1 != 0) and (action == left or action == right):
      action = right if action == left else left
  if (orientation & 2 != 0) and (action == up or action == down):
      action = up if action == down else down

  return action


@bottle.post('/move')
def move():
    data = bottle.request.json

    turn = data['turn']
    you = data['you']
    board = data['board']
    food = board['food']
    snakes = board['snakes']

    orientation = turn

    observations = prepareObservations(you, snakes, food, orientation)

    input = np.reshape(observations, (1, LAYER_WIDTH, LAYER_HEIGHT, NUM_LAYERS))
    prediction = model.predict(input, deterministic=True)
    output = prediction[0][0]

    direction = getDirection(output, orientation)

    return move_response(direction)


@bottle.post('/end')
def end():
    data = bottle.request.json

    """
    TODO: If your snake AI was stateful,
        clean up any stateful objects here.
    """
    print(json.dumps(data))

    return end_response()

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug=os.getenv('DEBUG', True)
    )
