# Turing Machine Pong

This project implements an AI to control a pong paddle using a modified Turing Machine architecture. What makes this different from a standard Turing Machine is as follows:

1. Separate input/output tapes
2. No left/right movement of head. Each cell in the tape is read sequentually from left to right.

The turing machine program is implemented via an evolutionary algorithm applied to a random population of programs to iteratively select the best performing individuals.

## Setup

Step 1: Create and activate the virtual environment
Mac/Linux

```
python3 -m venv tm_pong_env
source venv/bin/tm_pong_env
```

Windows

```
python -m venv tm_pong_env
venv\Scripts\tm_pong_env
```

Step 2: Install dependencies

```
pip install -r requirements.txt
```

Step 3: Run the project

```
python main.py
```
