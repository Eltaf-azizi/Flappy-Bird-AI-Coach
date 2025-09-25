
<h1 align="center">Flappy Bird AI Coach 🐦🤖</h1>

An interactive Flappy Bird game with an integrated AI coach powered by Reinforcement Learning (RL).
The goal of this project is to learn RL basics, integrate them into a game, and create an AI system that can give hints or adjust difficulty to help players improve.


## 🎯 Project Goals

 - Learn the fundamentals of reinforcement learning.
 - Integrate game environments with RL agents.
 - Build an AI “coach” that adapts to the player:
   - Give hints when the player struggles.
   - Adjust difficulty dynamically for balanced gameplay.


## Project Structure

    flappy-bird-ai-coach/
    ├── README.md                 # Project overview, setup, usage, learning goals
    ├── requirements.txt          # Python dependencies
    ├── .gitignore                # Ignore unnecessary files
    │
    ├── assets/                   # Game assets (sprites, backgrounds, sounds)
    │   ├── sprites/
    │   ├── sounds/
    │   └── fonts/
    │
    ├── config/                   
    │   ├── settings.py           # Game & training settings
    │   └── coach_config.py       # AI coach configuration
    │
    ├── src/
    │   ├── game/                 # Game logic (Pygame)
    │   ├── ai/                   # RL agent, environment, coach, training loop
    │   ├── integration/          # Play, train, evaluate scripts
    │   └── visualization/        # Training plots, demo recorder
    │
    ├── models/                   # Saved RL models
    ├── logs/                     # Training logs & coach feedback
    └── docs/                     # Documentation and notes


## ⚙️ Setup & Installation
1. Clone the Repository
```
git clone https://github.com/your-username/flappy-bird-ai-coach.git
cd flappy-bird-ai-coach
```
2. Create Virtual Environment (Optional but recommended)
```
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```
3. Install Dependencies
```
pip install -r requirements.txt
```


## ▶️ Usage
Play the Game with AI Coach

    python src/integration/play_with_coach.py

Train the RL Agent

    python src/integration/train_agent.py

Evaluate the Trained Agent

    python src/integration/evaluate_agent.py

Visualize Training Progress

    python src/visualization/plot_training.py


## 📊 Features

 - Playable Flappy Bird game built with Pygame.
 - RL Agent (DQN-based) to learn the game.
 - AI Coach that:
   - Gives feedback or hints.
   - Dynamically adjusts difficulty (pipe gap, speed).
 - Visualization tools for training metrics.
 - Logs & checkpoints for reproducibility.


## 📖 Learning Notes

This project is structured as both a fun game and a learning journey.
Inside the `docs/` folder:

 - `project_plan.md` → High-level goals and roadmap.
 - `rl_basics.md` → Key RL concepts applied here.
 - `future_improvements.md` → Ideas to expand the project.

## 🚀 Future Improvements

 - Add PPO or A2C agents for better training performance.
 - Implement a web-based version using Flask or Streamlit.
 - Add leaderboards and competitive training.
 - Extend AI coach with voice/text feedback.
