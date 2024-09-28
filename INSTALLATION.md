# Installation Guide for pyglotaran

This guide provides step-by-step instructions for installing **pyglotaran** on Windows, macOS, and Linux systems. It starts with installing pyglotaran if you already have Python installed, ensuring the use of virtual environments to keep your system clean. If you don't have Python installed yet, we'll guide you through installing Python on your system.

## Prerequisites

Before installing pyglotaran, ensure you have:

- **Python 3.10 or higher** installed on your system.
- Basic familiarity with command-line interfaces.

### Check if Python is Installed

Open your terminal or command prompt and run:

```shell
python --version
```

or

```
python3 --version
```

If Python is installed, this command will display the Python version number.

- If the version is 3.10 or higher, proceed to installing pyglotaran.
- If you see a lower version number or receive an error message, proceed to Installing Python on Your System.

## Installing pyglotaran

💡 _It is **recommended** to use a virtual environment to install pyglotaran into, to avoid possible conflicts with other Python packages and to keep your system Python environment clean._

### Step 1: Create a Project Directory

Create a new project folder. This could also be a folder downloaded from the [examples](https://github.com/glotaran/pyglotaran-examples).

### Step 2: Create a Virtual Environment

Create a virtual environment named venv.

For Windows:

```shell
python -m venv venv
```

<sub>(on MacOS/Linux) you may have to use `python3` instead of `python`.</sub>

### Step 3: Activate the Virtual Environment

Within your project folder run the following.

For Windows:

```shell
venv\Scripts\activate
```

For MacOS/Linux:

```shell
source venv/bin/activate
```

After activation, your command prompt should be prefixed with (venv).

### Step 4: Install pyglotaran

Install pyglotaran using pip

```shell
pip install pyglotaran
```

### Step 5: Verify the Installation

To verify that pyglotaran is installed correctly, run:

```shell
python -c "import pyglotaran; print(pyglotaran.__version__)"
```

If the installation was successful, this command will print the version number of pyglotaran.

## Installing Python on Your System

If you don't have Python installed, follow the instructions below for your operating system.

### Windows

#### Option 1: Install Python via the Microsoft Store

1. Open the Microsoft Store app.
2. Search for Python and select the latest version (Python 3.10 or higher).
3. Click Get or Install.

**\*Note**: This method installs Python only for the current user and doesn't require administrative privileges.\*

#### Option 2: Install Python from python.org

1. Go to the official Python website.
2. Download the latest Python installer (Python 3.10 or higher).
3. Run the installer:

- Check the box that says "Add Python to PATH".
- Choose "Install Now" for a default installation.

## macOS

#### Option 1: Install Python via Homebrew

1. Install Homebrew if you haven't already:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python:

```bash
brew install python
```

### Option 2: Install Python from python.org

1. Visit the official Python website.
2. Download the latest Python installer (Python 3.10 or higher).
3. Run the installer and follow the prompts.

## Linux

### Ubuntu/Debian

Update the package list:

```
sudo apt update
```

Install Python and pip:

```
sudo apt install python3 python3-venv python3-pip
```

### Other Distributions

Refer to your distribution's package manager documentation to install Python 3.10 or higher, python3-venv, and python3-pip.

# Next Steps

Now that you have Python and pyglotaran installed, you can start using pyglotaran for your data analysis.

## Using pyglotaran

pyglotaran is designed to be used within Python scripts or Jupyter Notebooks. It involves defining your analysis scheme, which includes:

- A model: Defines the kinetic scheme of your system.
- Parameters: Initial guesses and constraints for the model parameters.
- Experimental data: The data you want to fit.

You then use pyglotaran to optimize the model parameters to fit your data.

### Install Jupyter Notebook (Optional)

If you plan to use pyglotaran within Jupyter Notebooks, install Jupyter Notebook in your virtual environment:

```bash
pip install jupyterlab
```

Then, start the Jupyter Notebook server:

```bash
jupyter lab
```

This will open the Jupyter Notebook interface in your web browser.

## Examples and Tutorials

We have prepared comprehensive examples in the form of Jupyter Notebooks in the pyglotaran-examples repository. These examples illustrate how to use the framework for various use cases.

To get started:

1. Clone or [download](https://github.com/glotaran/pyglotaran-examples/archive/refs/heads/main.zip) the repository:

   ```bash
   git clone https://github.com/glotaran/pyglotaran-examples.git
   ```

2. Navigate to the example that best matches your use case.

3. Run the Jupyter Notebook to see pyglotaran in action.
