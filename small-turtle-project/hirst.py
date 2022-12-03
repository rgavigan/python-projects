# import colorgram
# rgb_colors = []
# colors = colorgram.extract('hirst_painting.jpeg', 30)
# for color in colors:
#     r = color.rgb.r
#     g = color.rgb.g
#     b = color.rgb.b
#     new_color = (r, g, b)
#     rgb_colors.append(new_color)
color_list = [(212, 149, 95), (215, 80, 62), (47, 94, 142), (231, 218, 92), (148, 66, 91), (22, 27, 40), (155, 73, 60), (122, 167, 195), (40, 22, 29), (39, 19, 15), (209, 70, 89), (192, 140, 159), (39, 131, 91), (125, 179, 141), (75, 164, 96), (229, 169, 183), (15, 31, 22), (51, 55, 102), (233, 220, 12), (159, 177, 54), (99, 44, 63), (35, 164, 196), (234, 171, 162), (105, 44, 39), (164, 209, 187), (151, 206, 220)]

import turtle as turtle_module
import random

turtle_module.colormode(255)
color_draw = turtle_module.Turtle()
color_draw.speed(0)
# Rows
color_draw.penup()
color_draw.setpos(-200, -200)
color_draw.pendown()
for i in range(10):
    # Columns
    for j in range(10):
        color_draw.dot(20, random.choice(color_list))
        color_draw.penup()
        color_draw.forward(50)
        color_draw.pendown()
    color_draw.penup()
    color_draw.backward(500)
    color_draw.left(90)
    color_draw.forward(50)
    color_draw.right(90)
    color_draw.pendown()


screen = turtle_module.Screen()
screen.exitonclick()
