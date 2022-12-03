import math

def ellipse_area(major, minor):
    '''
    Calculate the area of an ellipse
    Input: major and minor radius of ellipse
    Output: area of ellipse
    '''
    area = math.pi * major * minor
    return area
    
def rectangle_area(base, height):
    '''
    Calculate the area of a rectangle
    Input: rectangle base length and height length
    Output: area of rectangle
    '''
    area = base * height
    return area

def trapezoid_area(top, bottom, height):
    '''
    Calculate the area of a trapezoid
    Input: length of trapezoid's top, bottom, and height
    Output: area of trapezoid
    '''
    area = 1 / 2 * height * (top + bottom)
    return area