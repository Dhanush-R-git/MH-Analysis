import os
import pandas as pd  # type: ignore
from datetime import datetime
import inspect

# Define the path for the error log file
ERROR_LOG_FILE = "error_log.xlsx"

# Initialize the error log file if it doesn't exist
if not os.path.exists(ERROR_LOG_FILE):
    # Create an empty DataFrame with the required columns
    df = pd.DataFrame(columns=["File Name", "Error Type", "Error Name", "Code Line Number", "Debugged", "Status", "Timestamp"])
    df.to_excel(ERROR_LOG_FILE, index=False)

def log_error(file_name: str, error_type: str, error_name: str, line_number: int = None, debugged: str = "No", status: str = "Pending"):
    """
    Log an error or warning to the error log file.
    
    Args:
        file_name (str): Name of the file where the error occurred.
        error_type (str): Type of the error (e.g., "Error", "Warning").
        error_name (str): Description of the error or warning.
        line_number (int): Line number where the error occurred. If None, it will be auto-detected.
        debugged (str): Whether the error has been debugged ("Yes" or "No").
        status (str): Current status of the error (e.g., "Pending", "Resolved").
    """
    # Automatically detect the line number if not provided
    if line_number is None:
        frame = inspect.currentframe().f_back
        line_number = frame.f_lineno

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

def filter_errors_by_status(status: str) -> pd.DataFrame:
    """
    Filter errors in the log file by their status.

    Args:
        status (str): The status to filter by (e.g., "Pending", "Resolved").

    Returns:
        pd.DataFrame: A DataFrame containing the filtered errors.
    """
    if not os.path.exists(ERROR_LOG_FILE):
        print("Error log file does not exist.")
        return pd.DataFrame()

    # Load the existing error log
    df = pd.read_excel(ERROR_LOG_FILE)

    # Filter by status
    filtered_df = df[df["Status"] == status]

    print(f"Filtered errors with status '{status}':")
    print(filtered_df)

    return filtered_df

def update_error_status(file_name: str, error_name: str, debugged: str = "Yes", status: str = "Resolved"):
    """
    Update the "Debugged" and "Status" columns for a specific error.

    Args:
        file_name (str): Name of the file where the error occurred.
        error_name (str): Description of the error or warning.
        debugged (str): Whether the error has been debugged ("Yes" or "No").
        status (str): New status of the error (e.g., "Resolved", "Pending").
    """
    if not os.path.exists(ERROR_LOG_FILE):
        print("Error log file does not exist.")
        return

    # Load the existing error log
    df = pd.read_excel(ERROR_LOG_FILE)

    # Find the matching error and update its status
    mask = (df["File Name"] == file_name) & (df["Error Name"] == error_name)
    if not mask.any():
        print(f"No matching error found for file '{file_name}' with error '{error_name}'.")
        return

    df.loc[mask, "Debugged"] = debugged
    df.loc[mask, "Status"] = status

    # Save the updated log back to the Excel file
    df.to_excel(ERROR_LOG_FILE, index=False)

    print(f"Updated error '{error_name}' in file '{file_name}' to Debugged: {debugged}, Status: {status}.")