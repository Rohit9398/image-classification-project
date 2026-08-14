import runpy
import os

# Run app.py in the current directory
app_path = os.path.join(os.path.dirname(__file__), "app.py")
runpy.run_path(app_path, run_name="__main__")
