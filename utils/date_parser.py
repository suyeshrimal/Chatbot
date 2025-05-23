from datetime import datetime, timedelta
import dateparser

def extract_date(text):
    date_obj = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
    if date_obj:
        return date_obj.strftime("%Y-%m-%d")
    return None