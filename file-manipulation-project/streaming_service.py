class StreamingService:
    # Constructor that initializes three attributes
    def __init__(self, name, program_list, subscriber_list):
        """Initialize attributes"""
        self.name = name
        self.programs = program_list
        self.subscribers = subscriber_list

    # Five getter methods for the attributes + program/subscriber
    def get_name(self):
        """Retrieve name of streaming service"""
        return self.name

    def get_programs(self):
        """Retrieve list of Program objects in service"""
        return self.programs

    def get_subscribers(self):
        """Retrieve list of Subscriber objects in service"""
        return self.subscribers

    def get_program(self, title):
        """Retrieve program object if in the service"""
        for value in self.programs:
            if title == value.title:
                return value

    def get_subscriber(self, name):
        """Retrieve subscriber object if in the service"""
        for person in self.subscribers:
            if name == person.name:
                return person

    # Setter method for name of Program
    def set_name(self, new_name):
        """Set name of program"""
        self.name = new_name

    # Methods to add program/subscriber to service
    def add_program(self, program):
        """Append program to list of programs"""
        self.programs.append(program)

    def add_subscriber(self, subscriber):
        """Append subscriber to list of subscribers"""
        self.subscribers.append(subscriber)

    # Methods to remove program/subscriber from service
    def remove_program(self, title):
        """Within program list, if titles match up, remove the program from list"""
        for value in self.programs:
            if title == value.title:
                self.programs.remove(value)

    def remove_subscriber(self, name):
        """Within subscriber list, if names match up, remove the subscriber from list"""
        for person in self.subscribers:
            if name == person.name:
                self.subscribers.remove(person)

    # Methods to update program/subscriber objects
    def update_program(self, title, genre, creator, date):
        """Update program object in list of programs"""
        for value in self.programs:
            if title == value.title:
                if genre != '':
                    value.set_genre(genre)
                if creator != '':
                    value.set_creator(creator)
                if date != '':
                    value.set_release_date(date)

    def update_subscriber(self, name, userid, password):
        """Update program object in list of programs"""
        for person in self.subscribers:
            if name == person.name:
                if userid != '':
                    person.set_id(userid)
                if password != '':
                    person.set_password(password)

    # Function to sort program/subscriber lists
    def sort(self):
        """Sort program/subscriber lists"""
        title_list = []
        name_list = []
        # Fill lists with titles and names
        for value in self.programs:
            title_list.append(value.title)
        for person in self.subscribers:
            name_list.append(person.name)
        # Sort lists of titles and names
        title_list.sort()
        name_list.sort()
        # Create new instance of streaming service
        sorted_service = StreamingService(self.name, [], [])
        # Iteratively add programs and subscribers to service
        for title in title_list:
            sorted_program = self.get_program(title)
            sorted_service.programs.append(sorted_program)
        for name in name_list:
            sorted_subscriber = self.get_subscriber(name)
            sorted_service.subscribers.append(sorted_subscriber)
        return sorted_service

    def __repr__(self):
        # TODO: This is definitely broken
        return f'StreamingService("{self.name}", {self.programs}, {self.subscribers})'
