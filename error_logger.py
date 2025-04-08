import os
import pandas as pd # type: ignore
from datetime import datetime

# Define the path for the error log file
ERROR_LOG_FILE = "error_log.xlsx"

# Initialize the error log file if it doesn't exist
if not os.path.exists(ERROR_LOG_FILE):
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=["File Name", "Error Type", "Error Name", "Code Line Number", "Debugged", "Status", "Timestamp"])
    df.to_excel(ERROR_LOG_FILE, index=False)

def log_error(file_name: str, error_type: str, error_name: str, line_number: int, debugged: str = "No", status: str = "Pending"):
    """
    Log an error or warning to the error log file.
    
    Args:
        file_name (str): Name of the file where the error occurred.
        error_type (str): Type of the error (e.g., "Error", "Warning").
        error_name (str): Description of the error or warning.
        line_number (int): Line number where the error occurred.
        debugged (str): Whether the error has been debugged ("Yes" or "No").
        status (str): Current status of the error (e.g., "Pending", "Resolved").
    """
    # Load the existing error log
    df = pd.read_excel(ERROR_LOG_FILE)

    # Add the new error entry
    new_entry = {
        "File Name": file_name,
        "Error Type": error_type,
        "Error Name": error_name,
        "Code Line Number": line_number,
        "Debugged": debugged,
        "Status": status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    # Save the updated log back to the Excel file
    df.to_excel(ERROR_LOG_FILE, index=False)

    print(f"Logged {error_type}: {error_name} in {file_name} at line {line_number}")