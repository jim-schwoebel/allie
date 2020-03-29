import os
import tempfile
import string
import random


def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def temp_path_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokaggle')
    return path


def rand_temp_folder_generator():
    """Create and return a temporary directory with the path name '/temp_dir_name/autokeras' (E:g:- /tmp/autokeras)."""
    chars = string.ascii_uppercase + string.digits
    size = 6
    random_suffix = ''.join(random.choice(chars) for _ in range(size))
    sys_temp = temp_path_generator()
    path = sys_temp + '_' + random_suffix
    ensure_dir(path)
    return path
