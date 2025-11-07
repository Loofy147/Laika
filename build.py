import subprocess
import sys
import os

def run_command(command):
    """Runs a command and prints its output."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    """Main function."""
    # Create a virtual environment
    if not os.path.exists("venv"):
        run_command([sys.executable, "-m", "venv", "venv"])

    # Install dependencies
    run_command(["venv/bin/pip", "install", "-r", "ai_memory_system/requirements.txt"])

    # Run tests
    run_command(["venv/bin/python", "-m", "unittest", "ai_memory_system/tests.py"])

if __name__ == "__main__":
    main()
