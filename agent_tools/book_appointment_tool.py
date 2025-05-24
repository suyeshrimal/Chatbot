from langchain.tools import tool
from utils.validators import validate_inputs
from utils.date_parser import extract_date


@tool
def appointment_agent(name: str, phone: str, email: str, date: str) -> str:
    """
    Tool to book an appointment by collecting user name, phone, email, and preferred date.
    The tool validates inputs and extracts date in YYYY-MM-DD format.
    """
    valid, message = validate_inputs(name, phone, email)
    if not valid:
        return message

    formatted_date = extract_date(date)
    if not formatted_date:
        return "Invalid or unrecognized date format. Try 'next Monday' or '2025-06-01'."

    return f"Appointment booked for {name} on {formatted_date}. Contact: {email}, {phone}."
