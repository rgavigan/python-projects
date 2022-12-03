# Riley Gavigan - 251150776
from file_processing import build_new_service, update_service, write_update


def main():
    update_status = (input('Would you like to update a file? Y/N\n')).lower()
    while True:
        try:
            if update_status == 'y':
                file_name = input('Please enter the streaming service creation file (or \'done\' to exit):\n')
                if file_name.lower() == 'done':
                    print('Exiting program...')
                    exit(0)
                else:
                    service = build_new_service(file_name)
                    update_file = input('Please enter the update file you would like to read (or \'done\' to exit):\n')
                    if update_file.lower() == 'done':
                        print('Exiting program...')
                        exit(0)
                    else:
                        updated = update_service(update_file, service)
                        new_file = input('Please enter the name of the new file to be written:\n')
                        write_update(new_file, updated)
                update_status = (input('Would you like to enter another set of files? Y/N\n')).lower()
            elif update_status == 'n':
                print('Exiting program...')
                exit(0)
            else:
                raise ValueError('Invalid input')

        except ValueError as e:
            print(e)
            return 0


main()
