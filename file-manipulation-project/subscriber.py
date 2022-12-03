class Subscriber:
    # Constructor that initializes three attributes
    def __init__(self, name, userid, password):
        """Initialize attributes"""
        self.name = name
        self.id = userid
        self.password = password

    # Three getter methods for the attributes
    def get_name(self):
        """Retrieving subscriber name"""
        return self.name

    def get_id(self):
        """Retrieving subscriber userid"""
        return self.id

    def get_password(self):
        """Retrieving subscriber password"""
        return self.password

    # Three setter methods for the attributes
    def set_name(self, new_name):
        """Setting subscriber name"""
        self.name = new_name

    def set_id(self, new_userid):
        """Setting subscriber userid"""
        self.id = new_userid

    def set_password(self, new_password):
        """Setting subscriber password"""
        self.password = new_password

    # String representation of Subscriber
    def __repr__(self):
        """Creating string output of subscriber attributes"""
        return f'Subscriber("{self.name}", "{self.id}", "{self.password}")'
