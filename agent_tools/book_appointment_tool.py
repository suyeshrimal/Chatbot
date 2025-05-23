# from langchain.agents import tool

# import re
# import dateparser
# from langchain.tools import tool


# def validate_email(email):
#     return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


# def validate_phone(phone):
#     return bool(re.match(r"^\+?\d{10,15}$", phone))


# def parse_date(text):
#     parsed_date = dateparser.parse(text)
#     if parsed_date:
#         return parsed_date.strftime("%Y-%m-%d")
#     return None


# @tool
# def book_appointment_tool(name: str, email: str, phone: str, date: str) -> str:
#     """Tool to book an appointment with a user."""
#     if not validate_email(email):
#         return "Invalid email format."
#     if not validate_phone(phone):
#         return "Invalid phone number format."
#     if not dateparser.parse(date):
#         return "Invalid date format."

#     return f"Appointment booked for {name} on {date}.\nContact: {phone}, {email}"
def appointment_agent(name, phone, email, date):
    # This can be expanded to integrate Google Calendar, CRM, etc.
    return f"Appointment scheduled for {name} ({email}, {phone}) on {date}."