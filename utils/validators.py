import re

def validate_inputs(name, phone, email):
    if not name:
        return False, "Name is required."
    if not re.match(r"^[0-9]{10,15}$", phone):
        return False, "Invalid phone number. It should be 10–15 digits."
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email address."
    return True, "Valid"