import csv
from program import Program
from streaming_service import StreamingService
from subscriber import Subscriber


def build_new_service(data_file):
    """
    Function that reads data file with streaming service info
    Parameter: Data file with service data
    Output: StreamingService object from the information given
    """
    service = StreamingService('', [], [])
    try:
        with open(data_file) as csvfile:
            reader = csv.reader(csvfile)
            service.set_name(next(reader)[0])
            # Skip PROGRAMS
            next(reader)
            # Iterate through programs
            is_subscriber = False
            for row in reader:
                if row[0] == 'SUBSCRIBERS':
                    is_subscriber = True
                elif not is_subscriber:
                    new_program = Program(row[0], row[1], row[2], row[3])
                    service.add_program(new_program)
                    print(f'Adding program... {new_program.title}')
                else:
                    new_subscriber = Subscriber(row[0], row[1], row[2])
                    service.add_subscriber(new_subscriber)
                    print(f'Adding subscriber... {new_subscriber.name}')
        return service
    except FileNotFoundError:
        return None


def update_service(update_file, service):
    """
    Function that reads update file with changes
    and returns StreamingService object from information given
    Parameters: update file, StreamingService object
    Return: StreamingService object if file exists
    """
    try:
        with open(update_file) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the service name and PROGRAMS
            next(reader)
            next(reader)
            is_subscriber = False
            for row in reader:
                if row[0] == 'SUBSCRIBERS':
                    is_subscriber = True
                elif not is_subscriber:
                    # For all programs
                    if row[0] == '+':
                        add_program = Program(row[1], row[2], row[3], row[4])
                        service.add_program(add_program)
                        print(f'Adding program... {row[1]}')
                    elif row[0] == '-':
                        print(f'Removing program... {row[1]}')
                        service.remove_program(row[1])
                    elif row[0] == '^':
                        print(f'Updating program... {row[1]}')
                        service.update_program(row[1], row[2], row[3], row[4])
                elif is_subscriber:
                    # For all subscribers
                    if row[0] == '+':
                        add_subscriber = Subscriber(row[1], row[2], row[3])
                        service.add_subscriber(add_subscriber)
                        print(f'Adding subscriber... {row[1]}')
                    elif row[0] == '-':
                        service.remove_subscriber(row[1])
                        print(f'Removing subscriber... {row[1]}')
                    elif row[0] == '^':
                        service.update_subscriber(row[1], row[2], row[3])
                        print(f'Updating subscriber... {row[1]}')
        return service
    except FileNotFoundError:
        return None


def write_update(new_file, service):
    """
    Function that takes name of new file and streaming
    service object and writes update
    Parameters: name of file, StreamingService object
    """
    # Sort the updated file
    service = service.sort()
    with open(new_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        # Write service name and programs line
        writer.writerow(service.get_name().split('  '))
        writer.writerow(['PROGRAMS'])
        # Write each program object
        print(service.programs)
        for program in service.programs:
            writer.writerow([program.title, program.genre, program.creator, program.date])
        # Write subscribers header
        writer.writerow(['SUBSCRIBERS'])
        # Write each subscriber object
        for subscriber in service.get_subscribers():
            writer.writerow([subscriber.name, subscriber.id, subscriber.password])
