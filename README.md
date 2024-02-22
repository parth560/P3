# Reinforcement Learning Project


**Introduction**


In this project, I have implemented value iteration and Q-learning algorithms. These algorithms will be tested on various environments, including Gridworld, a simulated robot controller (Crawler), and Pacman.

**Getting Started
**

To begin, follow these steps:

1. Clone or download the project repository from GitHub.

2. Ensure you have Python 3.x installed on your machine.

3. Navigate to the project directory in your terminal or command prompt.

4. Run the autograder to test your solutions:

**Copy code
**

python autograder.py


**Note**: If your default Python interpreter refers to Python 2.7, use python3 autograder.py.


**Copy code**


python autograder.py


Run a specific question (e.g., q2):

Copy code

**python autograder.py -q q2**

Running Specific Tests
You can run specific tests by providing the path to the test case. For example:

**Copy code**


python autograder.py -t test_cases/q2/1-bridge-grid


**Making Modifications
**

Some modifications are required to avoid errors related to the cgi module. Follow these steps:

Replace import cgi with import html.
Modify cgi.escape(message) to html.escape(message).
