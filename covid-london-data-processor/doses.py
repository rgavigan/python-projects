# Riley Gavigan - 251150776
import csv
import re


def main():
    """
    Function that converts the dose data csv into a dictionary of lists in another csv
    Input: Prompts user for csv file to be read using csv module
    Output: Dictionary of lists with date = key, list of doses = value to doses.csv
    """

    # Initialize csv_file input and two dictionaries
    csv_file = input("Which data would you like to consider?\n")
    dose_dict = {}
    total_dose_dict = {}

    # Read data from input csv and append to dictionary
    with open(csv_file, 'r') as csvfile:
        dose_file = csv.reader(csvfile, delimiter=',')
        # Skip the first line of headings
        next(dose_file, None)
        # Iterate through file
        for row in dose_file:
            # Within each row, obtain the date for key and doses as a list for value
            date = row[0]
            doses = [cell for cell in row[1:6]]
            # Add date and doses into the dictionary dose_dict
            if date in dose_dict:
                dose_dict[date].append(doses)
            else:
                dose_dict[date] = doses

            # Obtain total doses through return value of function
            total_doses = count_doses_by_date(doses)
            # Add date and total doses into dictionary dose_dict
            if date in total_dose_dict:
                total_dose_dict[date].append(total_doses)
            else:
                total_dose_dict[date] = total_doses

    # Print the output of date and doses
    for key in dose_dict.keys():
        print(f"Date: {key}")
        print(f"Doses: {dose_dict[key]}")

    with open('doses.csv', 'w') as csvfile:
        for key in total_dose_dict.keys():
            csvfile.write(f"{key},{total_dose_dict[key]}\n")


def count_doses_by_date(dose_list):
    """
    Function that, for each date, adds number of vaccine doses, and outputs to doses.csv
    Input: The list of dose numbers for each location on given date
    Output: Total vaccination count for the given date [returned]
    """

    # Initialize total value for vaccination count
    total_value = 0
    for value in dose_list:
        # Deal with empty string values
        if value == '':
            value = 0
        # Substitute empty space for commas in values and add to total_value
        else:
            new_value = re.sub(',', '', value)
            total_value += int(new_value)
    return total_value


# Call the main function, which calls the function count_doses_by_date for each date
main()
