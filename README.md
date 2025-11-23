# Bat Echolocation Simulator (Spiking Neural Network)

Python/pygame simulator of a bat that navigates a 2D world using echolocation encoded as latency-coded spikes and processed by a small spiking neural network (snntorch/torch).

## Quickstart
- Install dependencies: `pip install -r requirements.txt`
- Run the simulator: `python bat_sim.py`
- Close the window to exit.

## What You’ll See
- 2D map with walls/circles (white), bat (blue), rays colored by distance (red=close, green=far), and ripple echoes at impact points.
- Raster overlays (top-left) showing input spike activity and motor spike activity per frame.

## Project Structure
- `bat_sim.py`: Environment, SpikingEar encoder, and BatBrain SNN plus the main loop.
- `requirements.txt`: Python dependencies.
- `SIMULATION_NOTES.md`: Deep dive on the math and logic behind sensing, encoding, and SNN control.

## Controls and Behavior
- The bat flies forward constantly.
- Motor spikes steer: right motor firing turns the bat right; left motor firing turns left.
- World wrapping keeps the bat inside the window.

## Dependencies
- Python 3.9+ recommended
- `pygame`, `torch`, `snntorch`

## Notes
- The SNN is pre-biased for obstacle avoidance (left sensors excite right motor, right sensors excite left motor).
- No training loop is required; all weights are initialized for reactive behavior.
