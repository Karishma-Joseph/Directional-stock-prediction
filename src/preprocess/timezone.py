import dateutil.parser
from dateutil import tz

# datestring = "2019-01-31T01:09:00.000+02:00"

def convert_timezone(datestring):
    """
    Input: datestring in ISO-8601 format. (Example: "2019-01-31T01:09:00.000+02:00")
    Output: returns the time converted to UTC 
    """
    date = dateutil.parser.parse(datestring)
    return (date - date.utcoffset()).replace(tzinfo=tz.tzutc())