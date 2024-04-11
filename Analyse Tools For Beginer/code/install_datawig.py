import os
import subprocess

def install_datawig():
    # Define the path to the repository
    repo_path = 'datawig'

    # Check if the directory exists and is not empty
    if not os.path.exists(repo_path) or not os.listdir(repo_path):
        # If the directory doesn't exist or is empty, clone the repository
        subprocess.check_call(['git', 'clone', 'https://github.com/awslabs/datawig'])

    # Change directory
    os.chdir(repo_path)

    # Change directory
    os.chdir('requirements')
    
    # Modify requirements.readthedocs.txt
    with open('requirements.readthedocs.txt', 'r') as file:
        lines = file.readlines()

    # Update the version of mxnet
    # Replace the old version with the new one
    for i, line in enumerate(lines):
        if 'mxnet' in line:
            lines[i] = 'mxnet>1.4.0\n'

    with open('requirements.readthedocs.txt', 'w') as file:
        file.writelines(lines)

    # Change directory back to repo_path
    os.chdir(os.path.join('..'))

    # Install the package
    subprocess.check_call(['pip', 'install', '.'])

# Call the function
install_datawig()
