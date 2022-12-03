# Riley Gavigan - 251150776
import csv


def minimum(vaccine_list):
    """
    Function that reads the data_list and determines the day with the least vaccinations
    Input: data_list
    Output: Tuple of (data, total_doses) for the day with the minimum total_doses
    """
    # Set a min_value at high number that will be used to compare dates
    min_value = 100000
    min_date = ''
    for date in vaccine_list:
        # Strip the number as there is whitespace, convert to integer
        vaccine_count = date[1]
        if vaccine_count < min_value:
            # Update max_value and max_date
            min_value = vaccine_count
            min_date = date[0]
    return min_date, min_value


def maximum(vaccine_list):
    """
    Function that reads the data_list and determines the day with the most vaccinations
    Input: data_list
    Output: Tuple of (data, total_doses) for the day with the maximum total_doses
    """
    # Set a max value at 0 that will be used to compare dates
    max_value = 0
    max_date = ''
    for date in vaccine_list:
        # Strip the number as there is whitespace, convert to integer
        vaccine_count = date[1]
        if vaccine_count > max_value:
            # Update max_value and max_date
            max_value = vaccine_count
            max_date = date[0

    return max_date, max_value


def total_vaccinations(vaccine_list):
    """
    Function that reads the data_list and adds up all vaccination counts
    Input: data_list
    Output: Integer of the total number of vaccinations over all days in doses.csv
    """
    # Initialize total_count to 0
    total_count = 0
    for date in vaccine_list:
        # Strip the number as there is whitespace, convert to integer
        vaccine_count = date[1]

        # Add the vaccine_count to the total_count
        total_count += vaccine_count

    return total_count


def main():
    """
    Function that reads doses.csv and converts into a list of tuples ('date', total_doses)
    Input: doses.csv file
    Output: Processed data using other functions and outputs it into vaccine_stats.txt
    """

    # Initialize data_list and open up doses.csv
    data_list = []
    with open('doses.csv', 'r') as csvfile:
        dose_reader = csv.reader(csvfile, delimiter=',')
        for row in dose_reader:
            date = row[0]
            total_doses = row[1].strip()
            # Create the list of tuples
            data_list.append((date, int(total_doses)))

    # Use other functions to obtain data that will be outputted
    minimum_day, minimum_value = minimum(data_list)
    maximum_day, maximum_value = maximum(data_list)
    total_vaccine_count = total_vaccinations(data_list)

    # Write the min, max, and total count of vaccines to vaccine_stats.txt
    with open('vaccine_stats.txt', 'w') as csvfile:
        csvfile.write(f"Total doses: {total_vaccine_count}\n")
        csvfile.write(f"Minimum date: {minimum_day}, Doses: {minimum_value}\n")
        csvfile.write(f"Maximum date: {maximum_day}, Doses: {maximum_value}\n")
        csvfile.write('')


# Call main
main()
