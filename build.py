# build.py
import os
import subprocess
import sys

PROJECT_DIR = "ai_memory_system"
VENV_DIR = os.path.join(PROJECT_DIR, "venv")

def create_virtualenv():
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Virtualenv already exists.")

def install_requirements():
    print("Installing requirements...")
    pip_exec = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "pip")
    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call([pip_exec, "install", "-r", req_file])
    else:
        print("requirements.txt not found. Please create it with dependencies.")

def prepare_project():
    if not os.path.exists(PROJECT_DIR):
        os.mkdir(PROJECT_DIR)
        print(f"Created project dir {PROJECT_DIR}")
    else:
        print(f"Project directory '{PROJECT_DIR}' already exists.")

def run_tests():
    # مثال على اختبار بسيط عبر سكريبت اختبارات
    test_module = f"{PROJECT_DIR}.tests"
    if os.path.exists(os.path.join(PROJECT_DIR, "tests.py")):
        print("Running tests...")
        python_exec = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
        subprocess.check_call([python_exec, "-m", test_module])
    else:
        print("No test script found.")

if __name__ == "__main__":
    prepare_project()
    create_virtualenv()
    install_requirements()
    run_tests()
    print("Build process completed.")
