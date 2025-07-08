import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Generates a detailed error message."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"

def CustomException(error: Exception, error_detail: sys) -> Exception:
    """Custom exception class that includes detailed error information."""
    error_message = error_message_detail(error, error_detail)
    return Exception(error_message)
