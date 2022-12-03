from turtle import Turtle, Screen
import random

# Draw a square
# square_turtle = Turtle()
# square_turtle.speed(0)
# for _ in range(4):
#     square_turtle.forward(100)
#     square_turtle.left(90)
#
# square_turtle.clear()

# Draw a dashed line
# dash_t = Turtle()
# dash_t.speed(0)
# for _ in range(10):
#     dash_t.forward(10)
#     dash_t.pu()
#     dash_t.forward(10)
#     dash_t.pd()
#
# dash_t.clear()

# Draw a bunch of different shapes
# diff_shapes = Turtle()
# diff_shapes.home()
# diff_shapes.speed(0)
# for i in range(3, 11):
#     num_sides = i
#     # Random colour
#     diff_shapes.color(random.random(), random.random(), random.random())
#     for j in range(num_sides):
#         diff_shapes.forward(100)
#         diff_shapes.right(360 / num_sides)
#
# diff_shapes.clear()

# Draw a random pathway turtle
# diff_path = Turtle()
# diff_path.pensize(15)
# diff_path.speed(0)
# for _ in range(250):
#     random_choice = random.randrange(1, 5)
#     diff_path.forward(25)
#     diff_path.color(random.random(), random.random(), random.random())
#     if random_choice == 1:
#         diff_path.right(90)
#     elif random_choice == 2:
#         diff_path.right(180)
#     elif random_choice == 3:
#         diff_path.right(270)
#     elif random_choice == 4:
#         diff_path.right(0)
#
# diff_path.clear()

# Spirograph turtle
spirograph = Turtle()
spirograph.speed(0)
for _ in range(90):
    spirograph.color(random.random(), random.random(), random.random())
    spirograph.circle(100)
    spirograph.right(4)


screen = Screen()
screen.exitonclick()
