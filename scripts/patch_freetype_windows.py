"""
scripts/patch_freetype_windows.py
Applies the Python 3.13 Windows fix to freetype-py:
1. Removes circular import (`import freetype` inside freetype/raw.py).
2. Explicitly resolves bundled `libfreetype.dll` via dirname(__file__).
3. Calls os.add_dll_directory() to prevent multi-minute find_library() PATH scan deadlock.
"""

from pathlib import Path
import sys

def patch_freetype():
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "freetype"
    raw_path = site_packages / "raw.py"
    if not raw_path.exists():
        print(f"freetype raw.py not found at {raw_path}")
        return False

    content = raw_path.read_text(encoding="utf-8")
    if "os.add_dll_directory" in content and "libfreetype.dll" in content and "import freetype\n" not in content:
        print("[patch_freetype] freetype-py is already patched and verified!")
        return True

    # Replacement
    old_block = """import freetype
from freetype.ft_types import *
from freetype.ft_enums import *
from freetype.ft_errors import *
from freetype.ft_structs import *

# First, look for a bundled FreeType shared object on the top-level of the
# installed freetype-py module.
system = platform.system()
if system == 'Windows':
    library_name = 'libfreetype.dll'
elif system == 'Darwin':
    library_name = 'libfreetype.dylib'
else:
    library_name = 'libfreetype.so'

filename = os.path.join(os.path.dirname(freetype.__file__), library_name)

# If no bundled shared object is found, look for a system-wide installed one.
if not os.path.exists(filename):
    # on windows all ctypes does when checking for the library
    # is to append .dll to the end and look for an exact match
    # within any entry in PATH.
    filename = ctypes.util.find_library('freetype')

    if filename is None:
        if platform.system() == 'Windows':
            # Check current working directory for dll as ctypes fails to do so
            filename = os.path.join(os.path.realpath('.'), "freetype.dll")
        else:
            filename = library_name

try:
    _lib = ctypes.CDLL(filename)"""

    new_block = """from freetype.ft_types import *
from freetype.ft_enums import *
from freetype.ft_errors import *
from freetype.ft_structs import *

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libfreetype.dll')
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(os.path.dirname(os.path.abspath(filename)))
    except Exception:
        pass

try:
    _lib = ctypes.CDLL(filename)"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        raw_path.write_text(content, encoding="utf-8")
        print("[patch_freetype] Successfully patched freetype/raw.py!")
        return True
    else:
        print("[patch_freetype] Warning: Expected exact unpatched block not found; checking state...")
        return False

if __name__ == "__main__":
    patch_freetype()
