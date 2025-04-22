# Function to read .bpmn file and return a single string

def read_bpmn_file(bpmn_path: str) -> str:
    """
    Reads a BPMN file and returns its content as a single string.

    Args:
        bpmn_path (str): Path to the BPMN file.

    Returns:
        str: The BPMN file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8 encoding.
        IOError: For general I/O errors while reading the file.
    """
    try:
        with open(bpmn_path, 'r', encoding='utf-8') as file:
            model_string = file.read()
        return model_string
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{bpmn_path}' was not found.")
    except UnicodeDecodeError:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Failed to decode BPMN file with UTF-8 encoding.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the BPMN file: {e}")

