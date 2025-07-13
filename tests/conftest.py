import sys
import os

# Add the project root directory to the Python search path
# This will allow imports like from src.module import ... to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
