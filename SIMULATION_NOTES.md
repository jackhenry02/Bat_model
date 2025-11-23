# Simulation Logic and Math

This document explains the bat echolocation simulator logic in detail: geometry, sensing, spike encoding, SNN computation, and control.

## Environment and Kinematics
- State: bat position `p = (x, y)` and heading `theta` (radians).
- Forward step each frame: `p <- p + v * [cos(theta), sin(theta)]` with constant speed `v = BAT_SPEED`.
- Heading updates when motor spikes bias one side: `theta <- theta + delta`, where `delta` is `+/- TURN_SPEED`. Screen wraps with `p.x mod width`, `p.y mod height`.

## Ray Casting (Echolocation Geometry)
- Number of rays: `N = 30` spanning a frontal cone with field-of-view `FOV = 90 deg` (`pi/2` rad).
- Ray `i` angle: `phi_i = theta - FOV/2 + (i / (N-1)) * FOV`.
- Direction vector per ray: `d_i = [cos(phi_i), sin(phi_i)]`.
- Rays march in fixed steps `s` until:
  - Hit a rectangle or circle obstacle; return distance `||p_hit - p||`.
  - Reach max range `R = 300`; return `R` and treat as "no hit".
- The hit point is used for drawing a ripple echo.

## Distance to Latency Encoding
- Distances become a "closeness" signal in `[0, 1]`:
  - `c_i = clamp((R - dist_i) / R, 0, 1)`.
- Latency coding via `snntorch.spikegen.latency` across `T = 30` timesteps:
  - Early spike for large `c_i` (near obstacle), late spike for small `c_i`.
  - No spike when `dist_i >= R` (masked out).
- Output: spike tensor `S_in[t, i]` where `t in [0, T-1]`, `i in [0, N-1]`.

## Spiking Neural Network (BatBrain)
- Layers:
  - Input: `N = 30` spike channels.
  - Hidden: Linear `fc1` -> LIF (Leaky Integrate-and-Fire) with decay `beta`.
  - Output: Linear `fc2` -> LIF producing 2 motor spikes: `[left_motor, right_motor]`.
- LIF dynamics (per neuron) follow `mem_t = beta * mem_{t-1} + input_t - spk_{t-1}` with thresholded spikes. `snntorch.snn.Leaky` handles this update.
- Forward unroll: for each timestep `t`, compute hidden spikes then motor spikes; collect all outputs as `S_out[t, k]` where `k in {0,1}`.

## Avoidance Bias (Pre-wiring)
- `fc2` weights are biased so that:
  - Left input half (rays pointing left) excites the right motor neuron.
  - Right input half excites the left motor neuron.
- This cross-coupling yields avoidance: obstacles on one side drive turning away.

## Motor Decision Rule
- Aggregate spikes over `T` timesteps:
  - `L = sum_t S_out[t, 0]`, `R = sum_t S_out[t, 1]`.
- Turn logic:
  - If `L > 1.1 * R`, turn left (negative delta).
  - If `R > 1.1 * L`, turn right (positive delta).
  - Otherwise keep heading.
- The bat still advances forward every frame; turning adjusts only the heading.

## Visualization
- Rays colored by distance: red (near) to green (far). Echo ripple at hit point with radius increasing as distance shortens.
- Raster overlays:
  - Input panel: frame-summed input spikes per ray.
  - Motor panel: frame-summed motor spikes (2 neurons).
- Overlays are drawn directly onto the pygame surface for real-time feedback.

## Parameters (default)
- `NUM_RAYS = 30`, `FOV = 90 deg`, `RAY_MAX_DIST = 300`, `BAT_SPEED = 2.5 px/frame`, `TURN_SPEED = 3 deg/frame`, `T = 30` timesteps, `beta = 0.9`.

## Extending
- Adjust `NUM_RAYS`/`FOV` for wider or narrower sensing.
- Change `beta` or hidden size for different temporal integration.
- Replace bias-only weights with online Hebbian updates for adaptive behavior.
