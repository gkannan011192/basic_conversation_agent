from datetime import datetime
from typing import Dict, Optional
from ava_agent.core.schedules import (
    FRIDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
    THURSDAY_SCHEDULE
)

class ScheduleContextGenerator:
    
    SCHEDULES={
        0: SUNDAY_SCHEDULE,
        1: MONDAY_SCHEDULE,
        2: TUESDAY_SCHEDULE,
        3: WEDNESDAY_SCHEDULE,
        4: THURSDAY_SCHEDULE,
        5: FRIDAY_SCHEDULE,
        6: SATURDAY_SCHEDULE 
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> tuple[datetime.time, datetime.time]:
        """Parse a time range string in the format "HH:MM-HH:MM" into start and end times.
        Args:
            time_range (str): A string representing a time range in "HH:MM-HH:MM" format (e.g. "09:00-17:00")
        Returns:
            tuple[datetime.time, datetime.time]: A tuple containing:
                - start_time: The parsed start time as datetime.time
                - end_time: The parsed end time as datetime.time
        Raises:
            ValueError: If the time_range string is not in the correct format
        """
        try:
            start_str, end_str = time_range.split("-")
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            return start_time, end_time
        except Exception as e:
            raise ValueError("Time_range is not in the correct format")

    @classmethod
    def get_current_activity(client) -> Optional[str]:
        """
        Retrieves the current activity from client's schedule based on the current time and day.
        This method checks the client's schedule for the current day and determines if any
        activity is scheduled for the current time. It handles time ranges that may span across
        midnight (where end time is less than start time).
        Args:
            client: The client object containing the SCHEDULES attribute and _parse_time_range method.
        Returns:
            Optional[str]: The current activity if one is scheduled for the current time,
                          None if no activity is scheduled.
        Example:
            If current time is 14:30 and there's a schedule entry "14:00-15:00": "Meeting",
            this method will return "Meeting".
        """
        current_datetime=datetime.now()
        current_time=current_datetime.time()
        current_day=current_datetime.weekday()
        
        schedule=client.SCHEDULES.get(current_day, {})
        for time_range, activity in schedule.items():
            start_time,end_time=client._parse_time_range(time_range)
            if start_time > end_time:
                if current_time>=start_time or current_time<=end_time:
                    return activity
            else:
                if start_time<=current_time<=end_time:
                    return activity
        return None

    @classmethod
    def get_schedule_for_day(client, day:int) -> Dict[str,str]:
        """
        Retrieves the schedule for a specific day from the client's schedules.
        Args:
            client: The client object containing schedule information
            day (int): The day number to get the schedule for
        Returns:
            Dict[str,str]: A dictionary containing the schedule for the specified day.
                           Returns empty dict if no schedule exists for that day.
        """
        return client.SCHEDULES.get(day,{})
            
    

