# Riley Gavigan - 251150776

# Import the area calculation functions
import area_calculation

def is_valid_shape(in_value):
    '''
    Determine if input is a valid shape
    Input: Shape name provided by user
    Output: True if valid, False if invalid
    '''
    if in_value.lower() == 'ellipse' or in_value.lower() == 'rectangle' or in_value.lower() == 'trapezoid':
        return True
    else:
        return False


def main():
    # Create empty list
    area_list = []
    while True:
        # Retrieve input shape
        in_value = input('What shape would you like to calculate?')
        if in_value == 'done':
            # Print out list o4f areas and end loop
            area_list.sort()
            print(area_list)
            return False

        elif not is_valid_shape(in_value):
            # Print out message and prompt user again
            print('Sorry, but that shape doesn\'t exist in our system. Please try again.')

        else:

            if in_value.lower() == 'ellipse':
                # Take major and minor values as inputs
                major = float(input('What is the major radius of the ellipse?'))
                minor = float(input('What is the minor radius of the ellipse?'))
                # Calculate ellipse area using imported function
                ellipse_area = area_calculation.ellipse_area(major, minor)
                # Print area and append area to list
                print(f'The calculated area is {ellipse_area:.2f}')
                area_list.append(round(ellipse_area, 2))

            elif in_value.lower() == 'rectangle':
                # Take base and height as inputs
                base = float(input('What is the length of the rectangle\'s base'))
                rectangle_height = float(input('What is the height of the rectangle?'))
                # Calculate rectangle area using imported function
                rectangle_area = area_calculation.rectangle_area(base, rectangle_height)
                # Print area and append area to list
                print(f'The calculated area is {rectangle_area:.1f}')
                area_list.append(round(rectangle_area, 2))

            else:
                # Take top, bottom, and height as inputs
                top = float(input('What is the length of the trapezoid\'s top?'))
                bottom = float(input('What is the length of the trapezoid\'s bottom?'))
                trapezoid_height = float(input('What is the trapezoid\'s height?'))
                # Calculate trapezoid area using imported function
                trapezoid_area = area_calculation.trapezoid_area(top, bottom, trapezoid_height)
                # Print area and append to list
                print(f'The calculated area is {trapezoid_area:.1f}')
                area_list.append(round(trapezoid_area, 2))
                
main()