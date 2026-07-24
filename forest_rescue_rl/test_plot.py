"""Quick 200-ep training to generate curves."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forest_rescue_rl.train import train
train(n_episodes=200, log_interval=20)
