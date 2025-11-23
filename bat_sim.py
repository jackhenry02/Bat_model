import math
import random
from collections import deque
from typing import List, Tuple

import pygame
import torch
import snntorch as snn
from snntorch import spikegen


# Simulation constants
SCREEN_WIDTH, SCREEN_HEIGHT = 900, 600
BG_COLOR = (18, 18, 18)
BAT_COLOR = (70, 140, 255)
OBSTACLE_COLOR = (240, 240, 240)
RAY_MAX_DIST = 300
NUM_RAYS = 30
FOV_DEG = 90
BAT_SPEED = 2.5
TURN_SPEED = math.radians(3.0)
SPIKE_TIME_STEPS = 30  # total steps per frame for SNN processing


class Environment:
    """2D world with simple ray casting for echolocation."""

    def __init__(self, width: int, height: int, num_rays: int, fov_deg: float, max_dist: float):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bat Echolocation Simulator")
        self.width, self.height = width, height
        self.num_rays = num_rays
        self.fov = math.radians(fov_deg)
        self.max_dist = max_dist
        self.clock = pygame.time.Clock()

        # Obstacles: mix of rectangles and circles
        self.rects = [
            pygame.Rect(150, 100, 120, 40),
            pygame.Rect(500, 180, 60, 200),
            pygame.Rect(300, 400, 180, 40),
            pygame.Rect(100, 250, 60, 150),
        ]
        self.circles = [
            ((700, 120), 40),
            ((650, 380), 55),
            ((200, 500), 35),
        ]

        # Agent state
        self.bat_pos = pygame.Vector2(width * 0.2, height * 0.5)
        self.bat_theta = 0.0

    def step_forward(self):
        direction = pygame.Vector2(math.cos(self.bat_theta), math.sin(self.bat_theta))
        self.bat_pos += direction * BAT_SPEED
        # Keep bat within bounds by wrapping
        self.bat_pos.x %= self.width
        self.bat_pos.y %= self.height

    def rotate(self, delta_theta: float):
        self.bat_theta += delta_theta

    def _ray_hit(self, origin: pygame.Vector2, direction: pygame.Vector2) -> Tuple[float, Tuple[float, float]]:
        """Cast a ray until hit or max distance; returns (distance, hit_point)."""
        step = 4.0
        current = origin.copy()
        for i in range(int(self.max_dist // step)):
            current += direction * step
            # Rect collision
            for rect in self.rects:
                if rect.collidepoint(current):
                    return current.distance_to(origin), (current.x, current.y)
            # Circle collision
            for center, radius in self.circles:
                if pygame.Vector2(center).distance_to(current) <= radius:
                    return current.distance_to(origin), (current.x, current.y)
        # No collision
        return self.max_dist, (origin.x + direction.x * self.max_dist, origin.y + direction.y * self.max_dist)

    def cast_rays(self) -> Tuple[List[float], List[Tuple[float, float]]]:
        half_fov = self.fov / 2
        base_angle = self.bat_theta - half_fov
        distances = []
        hits = []
        for i in range(self.num_rays):
            angle = base_angle + (i / (self.num_rays - 1)) * self.fov
            direction = pygame.Vector2(math.cos(angle), math.sin(angle))
            dist, hit_pt = self._ray_hit(self.bat_pos, direction)
            distances.append(dist)
            hits.append(hit_pt)
        return distances, hits

    def _ray_color(self, dist: float) -> Tuple[int, int, int]:
        # Close -> red, Far -> green
        t = max(0.0, min(1.0, dist / self.max_dist))
        r = int(255 * (1 - t))
        g = int(200 * t)
        return (r, g, 60)

    def draw(self, distances: List[float], hit_points: List[Tuple[float, float]]):
        self.screen.fill(BG_COLOR)

        # Obstacles
        for rect in self.rects:
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect, border_radius=4)
        for center, radius in self.circles:
            pygame.draw.circle(self.screen, OBSTACLE_COLOR, center, radius, width=2)

        # Rays and echoes
        half_fov = self.fov / 2
        base_angle = self.bat_theta - half_fov
        for i in range(self.num_rays):
            angle = base_angle + (i / (self.num_rays - 1)) * self.fov
            direction = pygame.Vector2(math.cos(angle), math.sin(angle))
            end_point = self.bat_pos + direction * distances[i]
            color = self._ray_color(distances[i])
            pygame.draw.line(self.screen, color, self.bat_pos, end_point, width=2)
            # Echo ripple
            if distances[i] < self.max_dist:
                ripple_radius = 5 + int((self.max_dist - distances[i]) * 0.05)
                pygame.draw.circle(self.screen, color, (int(hit_points[i][0]), int(hit_points[i][1])), ripple_radius, width=1)

        # Bat body
        pygame.draw.circle(self.screen, BAT_COLOR, (int(self.bat_pos.x), int(self.bat_pos.y)), 10)
        # Nose direction indicator
        nose = pygame.Vector2(math.cos(self.bat_theta), math.sin(self.bat_theta)) * 15
        pygame.draw.line(self.screen, (255, 255, 255), self.bat_pos, self.bat_pos + nose, width=2)

    def tick(self, fps: int = 60):
        return self.clock.tick(fps)


class SpikingEar:
    """Convert ray distances to latency-coded spike trains."""

    def __init__(self, max_dist: float, num_steps: int):
        self.max_dist = max_dist
        self.num_steps = num_steps

    def encode(self, distances: List[float]) -> torch.Tensor:
        dist_tensor = torch.tensor(distances, dtype=torch.float32)
        closeness = (self.max_dist - dist_tensor).clamp(min=0.0, max=self.max_dist) / self.max_dist
        spikes = spikegen.latency(closeness.unsqueeze(0), num_steps=self.num_steps)[0]  # (num_steps, num_inputs)
        no_hit_mask = dist_tensor >= self.max_dist
        spikes[:, no_hit_mask] = 0.0  # drop spikes if nothing was detected
        return spikes


class BatBrain(torch.nn.Module):
    """Simple feedforward SNN with avoidance bias."""

    def __init__(self, num_inputs: int, hidden_size: int = 64, beta: float = 0.9):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_inputs, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = torch.nn.Linear(hidden_size, 2)
        self.lif2 = snn.Leaky(beta=beta)
        self._init_weights(num_inputs)

    def _init_weights(self, num_inputs: int):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)

        # Encourage avoidance: left sensors excite right motor and vice versa
        half = num_inputs // 2
        with torch.no_grad():
            for i in range(half):
                self.fc2.weight[1, i] += 0.8  # left inputs -> right motor
            for i in range(half, num_inputs):
                self.fc2.weight[0, i] += 0.8  # right inputs -> left motor

    def forward(self, spike_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # spike_seq: [T, num_inputs]
        num_steps = spike_seq.shape[0]
        syn1 = self.fc1.weight.new_zeros((self.fc1.weight.size(0)))
        mem1 = self.fc1.weight.new_zeros((self.fc1.weight.size(0)))
        syn2 = self.fc2.weight.new_zeros((self.fc2.weight.size(0)))
        mem2 = self.fc2.weight.new_zeros((self.fc2.weight.size(0)))

        hidden_spikes = []
        out_spikes = []
        for t in range(num_steps):
            cur_input = spike_seq[t]
            h = self.fc1(cur_input)
            spk1, mem1 = self.lif1(h, mem1)
            h2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(h2, mem2)
            hidden_spikes.append(spk1)
            out_spikes.append(spk2)
        return torch.stack(out_spikes), torch.stack(hidden_spikes), mem2


def draw_raster(surface: pygame.Surface, spikes: deque, pos: Tuple[int, int], size: Tuple[int, int], color=(120, 220, 120)):
    """Simple raster plot overlay onto pygame surface."""
    x, y = pos
    w, h = size
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h))
    if len(spikes) == 0:
        return
    # spikes holds tensors of shape [num_inputs]
    num_neurons = spikes[0].numel()
    for t, spk_vec in enumerate(spikes):
        px = x + t
        if px >= x + w:
            break
        for n in range(num_neurons):
            if spk_vec[n] > 0:
                py = y + int((n / num_neurons) * h)
                surface.fill(color, (px, py, 1, 2))


def main():
    env = Environment(SCREEN_WIDTH, SCREEN_HEIGHT, NUM_RAYS, FOV_DEG, RAY_MAX_DIST)
    ear = SpikingEar(RAY_MAX_DIST, SPIKE_TIME_STEPS)
    brain = BatBrain(NUM_RAYS)

    running = True
    input_history = deque(maxlen=200)   # aggregate spikes per frame for raster
    motor_history = deque(maxlen=200)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        distances, hit_points = env.cast_rays()
        spike_train = ear.encode(distances)  # [T, num_inputs]
        out_spikes, hidden_spikes, _ = brain(spike_train)

        # Motor decision: more spikes on one side -> turn opposite
        left_motor = out_spikes[:, 0].sum().item()
        right_motor = out_spikes[:, 1].sum().item()
        if left_motor > right_motor * 1.1:
            env.rotate(-TURN_SPEED)
        elif right_motor > left_motor * 1.1:
            env.rotate(TURN_SPEED)

        env.step_forward()

        # Draw world
        env.draw(distances, hit_points)

        # Raster overlay
        # Compress time dimension by max over time for clarity
        frame_input_spikes = spike_train.sum(dim=0)  # [num_inputs]
        frame_motor_spikes = out_spikes.sum(dim=0)   # [2]
        input_history.append(frame_input_spikes)
        motor_history.append(frame_motor_spikes)

        draw_raster(env.screen, input_history, pos=(10, 10), size=(200, 120), color=(220, 180, 80))
        draw_raster(env.screen, motor_history, pos=(10, 140), size=(200, 40), color=(120, 220, 120))
        pygame.draw.rect(env.screen, (200, 200, 200), (10, 10, 200, 120), 1)
        pygame.draw.rect(env.screen, (200, 200, 200), (10, 140, 200, 40), 1)

        pygame.display.flip()
        env.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
