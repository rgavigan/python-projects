class Program:
    # Constructor that initializes four attributes
    def __init__(self, title, genre, creator, date):
        """Initializing attributes"""
        self.title = title
        self.genre = genre
        self.creator = creator
        self.date = date

    # Four getter methods for the attributes
    def get_title(self):
        """Retrieving program title"""
        return self.title

    def get_genre(self):
        """Retrieving program genre"""
        return self.genre

    def get_creator(self):
        """Retrieving program creator"""
        return self.creator

    def get_release_date(self):
        """Retrieving program release date"""
        return self.date

    # Four setter methods for the attributes
    def set_title(self, new_title):
        """Setting program title"""
        self.title = new_title

    def set_genre(self, new_genre):
        """Setting program genre"""
        self.genre = new_genre

    def set_creator(self, new_creator):
        """Setting program creator"""
        self.creator = new_creator

    def set_release_date(self, new_date):
        """Setting program release date"""
        self.date = new_date

    # String representation of Program
    def __repr__(self):
        """Creating string output of program attributes"""
        return f'Program("{self.title}", "{self.genre}", "{self.creator}", "{self.date}")'
