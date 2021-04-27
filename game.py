import math
import random

import pygame
from pygame import mixer

# Intialize the pygame
pygame.init()

# create the screen
screen = pygame.display.set_mode((800, 600))

# Background
background = pygame.image.load('background.png') # Put relevant bg pic

# Sound
mixer.music.load("background.wav")
mixer.music.play(-1)

# Caption and Icon
pygame.display.set_caption("Space Invader")  # Change accordingly
icon = pygame.image.load('ufo.png')
pygame.display.set_icon(icon)

# Player
playerImg = pygame.image.load('player.png')  # put relevant pic for player
playerX = ?
playerY = ?
playerX_change = 0
playerY_change = 

# Enemy
enemyImg = []
enemyX = []
enemyY = []
enemyX_change = []
enemyY_change = []
num_of_enemies = 6
num_of_frames = 3

for i in range(num_of_enemies):
    enemyImg.append([pygame.image.load('enemy_frame1.png'),pygame.image.load('enemy_frame2.png'),pygame.image.load('enemy_3.png')])
    enemyX.append(random.randint(0, 736))   # modify range
    enemyY.append(random.randint(50, 150))  # modify range
    enemyX_change.append(4)
    enemyY_change.append(40)

# Bullet

# Ready - You can't see the bullet on the screen
# Fire - The bullet is currently moving

bulletImg = pygame.image.load('bullet.png')  # Put bullet pic
bulletX = 0
bulletY = 480
bulletX_change = 10?
bulletY_change = 10?
bullet_state = "ready"

# Score

score_value = 0
font = pygame.font.Font('freesansbold.ttf', 32)

textX = 10
testY = 10

# Game Over
over_font = pygame.font.Font('freesansbold.ttf', 64)


def show_score(x, y):
    score = font.render("Score : " + str(score_value), True, (255, 255, 255))
    screen.blit(score, (x, y))


def game_over_text():
    over_text = over_font.render("GAME OVER", True, (255, 255, 255))
    screen.blit(over_text, (200, 250))


def player(x, y, angle):
    rotated_player = pygame.transform.rotate(playerImg, angle)
    screen.blit(rotated_player, (x, y))


def enemy(x, y, i, j): # j is for the illusion. Multiple frames
    screen.blit(enemyImg[i][j], (x, y))


def fire_bullet(x, y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x + ?, y + ?)) # replace the ?

# Print one circle to specify the player's shooting range : Radius R
def circle(x, y, R):
    BLUE = (0, 0, 255)
    pygame.draw.circle(screen, BLUE, (x,y), R)

def calc_distance(x1, y1, x2, y2):
    distance = math.sqrt(math.pow(x1 - x2, 2) + (math.pow(y1 - y2, 2)))
    return distance

def isCollision(enemyX, enemyY, bulletX, bulletY):
    distance = math.sqrt(math.pow(enemyX - bulletX, 2) + (math.pow(enemyY - bulletY, 2)))
    if distance < 27: # Change distance
        return True
    else:
        return False

j = 1
# Game Loop
running = True
while running:

    # RGB = Red, Green, Blue
    screen.fill((0, 0, 0))
    # Background Image
    screen.blit(background, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # if keystroke is pressed check whether its right or left
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:   # Change bullet from space to phone button
                if bullet_state is "ready":
                    bulletSound = mixer.Sound("laser.wav")
                    bulletSound.play()
                    # Get the current x cordinate of the spaceship
                    bulletY = ? # Set bullet y's value based on player's position
                    bulletX = ? # Set bullet x's value based on player's position
                    bullet_startingY = bulletY
                    bullet_startingX = bulletX
                    fire_bullet(bulletX, bulletY)

    # 5 = 5 + -0.1 -> 5 = 5 - 0.1
    # 5 = 5 + 0.1

    playerX = x_from_kalman
    playerY = y_from_kalman
    if playerX <= 0:  # Set limits, modify values accordingly
        playerX = 0
    elif playerX >= 736:
        playerX = 736 # Similarly do for y also
        
    draw_circle(playerX, playerY, R)  # Displaying circle for player's range 

    # Enemy Movement
    for i in range(num_of_enemies):

        # Collision
        collision = isCollision(enemyX[i], enemyY[i], bulletX, bulletY)
        if collision:
            explosionSound = mixer.Sound("explosion.wav")
            explosionSound.play()
            bullet_state = "ready"
            score_value += 1
            enemyX[i] = random.randint(0, 736)
            enemyY[i] = random.randint(50, 150)

        enemy(enemyX[i], enemyY[i], i, j)
        j += 1
        if j == 4: j=1
    
    # Bullet Movement
    if calc_distance(bulletX, bulletY, bullet_startingX, bullet_startingY ) >= R: # R is radius of range
        bullet_state = "ready"
    
    theta = ? # Get orientation from UDP, covert theta to degrees
    if theta > +60: theta = 60
    elif theta < -60: theta = -60
    
    bulletY_change , bulletX_change = ? # calculate using theta
    if bullet_state is "fire":
        fire_bullet(bulletX, bulletY)
        bulletY += bulletY_change
        bulletX += bulletX_change

    player(playerX, playerY, theta)
    show_score(textX, testY)
    pygame.display.update()
