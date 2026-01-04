# File: pogogo/lagged/agent.py

import copy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def per_state_sinkhorn(
    actor,
    ref_policy,
    states,
    K: int = 4,
    blur: float = 0.05,
    p: int = 2,
    use_ref_grad: bool = False,
    backend: str = "tensorized",
    sinkhorn_loss=None,  # Optional pre-instantiated SamplesLoss
):
    """
    Per-state Sinkhorn (W_p) between actor(·|s) and ref_policy(·|s).
    - states: [B, s_dim]
    - For each state s, we draw K samples from both policies
      via different z ~ N(0, I) and compute Sinkhorn distance.
    Implementation notes:
    - Fully vectorized: [B, K, d] vs [B, K, d] -> GeomLoss
    - GeomLoss old versions: no 'reduction' arg, so we don't use it.
    """
    B = states.size(0)
    a_dim = actor.head.out_features
    dev = states.device

    # z for both policies: [B, K, a_dim]
    z_a = torch.randn(B, K, a_dim, device=dev)
    z_b = torch.randn(B, K, a_dim, device=dev)

    # Tile states: [B, s_dim] -> [B, K, s_dim] -> [B*K, s_dim]
    states_tiled = states.unsqueeze(1).expand(B, K, states.size(1))
    states_flat = states_tiled.reshape(B * K, -1)

    # Actor samples: [B*K, a_dim] -> [B, K, a_dim]
    a = actor(states_flat, z_a.reshape(B * K, a_dim)).reshape(B, K, a_dim)

    # Reference policy samples
    with torch.set_grad_enabled(use_ref_grad):
        b = ref_policy(states_flat, z_b.reshape(B * K, a_dim)).reshape(B, K, a_dim)

    # GeomLoss batch call (no reduction arg)
    if sinkhorn_loss is None:
        sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend=backend)
    else:
        sinkhorn = sinkhorn_loss
    loss = sinkhorn(a, b)  # expects [B, K, d] vs [B, K, d]

    # 일부 버전에서 scalar가 아니라 벡터가 나올 수 있으므로 방어적으로 평균
    if loss.dim() > 0:
        loss = loss.mean()

    return loss


class Actor(nn.Module):
    """
    Transport Map Actor Network
    Implements T_s: z ~ N(0,I) → action space
    """

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        # Transport map: (state, z) → action
        self.l1 = nn.Linear(action_dim + state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, state, z):
        # state: [B, s_dim], z: [B, a_dim]
        x = torch.cat([state, z], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return torch.tanh(self.head(x)) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class POGO:
    """
    Unified POGO: multiple actors trained in a JKO chain.

    - Critic: uses the first actor (index 0) as 'behavior policy' for target actions.

    - Actor training:
      * actor[0] (π_0): POGO style, L2 to dataset actions
      * actor[i] (π_i, i >= 1): Sinkhorn W2 to actor[i-1] (POGO_Refine style)

    Args:
        w2_weights: list of W2 weights, one per actor (len = num_actors).
                    e.g. [0.2, 0.2] (two actors) or [0.2, 0.2, 0.3] (three actors)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        w2_weights=[1.0, 1.0],
        lr=3e-4,
        freeze_critic=False,
        noise_shaping=True,
    ):
        self.num_actors = len(w2_weights)
        assert self.num_actors >= 1, "At least one actor is required."

        self.w2_weights = w2_weights

        # Multiple actors
        self.actors = []
        self.actor_targets = []
        self.actor_optimizers = []

        for _ in range(self.num_actors):
            actor = Actor(state_dim, action_dim, max_action).to(device)
            actor_target = Actor(state_dim, action_dim, max_action).to(device)
            actor_target.load_state_dict(actor.state_dict())
            
            optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_optimizers.append(optimizer)

        # Critic: uses the first actor as behavior policy for target actions
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.noise_shaping = noise_shaping
        self.total_it = 0
        # Cache SamplesLoss object to avoid repeated instantiation
        self._sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")

    # ------------------------------------------------------------------
    # Action selection API (eval에서 actor_idx로 선택)
    # ------------------------------------------------------------------

    def select_action(self, state, deterministic: bool = True, actor_idx: Optional[int] = None):
        """
        Select action using transport map.

        Args:
            state: 1D or 2D numpy array (state vector)
            deterministic: if True, z = 0; else z ~ N(0, I)
            actor_idx: which actor to use (None → last actor)
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)

        if actor_idx is None:
            actor_idx = self.num_actors - 1  # default: use last actor

        actor = self.actors[actor_idx]

        if deterministic:
            z = torch.zeros(state.shape[0], actor.head.out_features, device=device)
        else:
            z = torch.randn(state.shape[0], actor.head.out_features, device=device)

        action = actor(state, z)
        return action.detach().cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, replay_buffer, batch_size=256):
        """
        Unified training:
        - Critic: uses the first actor as behavior policy for TD target.
        - All actors: updated sequentially with respective JKO losses.
        """

        self.total_it += 1

        # Sample batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ------------------------
        # Critic training
        # ------------------------
        if not self.freeze_critic:
            with torch.no_grad():
                # Use target actor for TD3 stability
                self.actor_targets[0].eval()
                z_target = torch.randn_like(action, device=device)
                next_action = self.actor_targets[0](next_state, z_target)

                # Base TD target
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

                # Optional TD3-style noise shaping on target actions
                if self.noise_shaping:
                    action_noise = self.policy_noise * torch.randn_like(action)
                    action_noise = action_noise.clamp(-self.noise_clip, self.noise_clip)
                    action_noise = action_noise * self.max_action
                    noisy_next_action = (next_action + action_noise).clamp(
                        -self.max_action, self.max_action
                    )
                    noise_target_Q1, noise_target_Q2 = self.critic_target(
                        next_state, noisy_next_action
                    )
                    noise_target_Q = reward + not_done * self.discount * torch.min(
                        noise_target_Q1, noise_target_Q2
                    )
                    target_Q = torch.min(target_Q, noise_target_Q)

                self.actor_targets[0].train()

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            metrics = {"critic_loss": float(critic_loss.item())}
        else:
            # Critic is frozen, don't log critic_loss
            metrics = {}

        # ------------------------
        # Actor training
        # ------------------------
        if self.total_it % self.policy_freq == 0:
            # All actors in train mode
            for actor in self.actors:
                actor.train()

            for i in range(self.num_actors):
                actor_i = self.actors[i]
                w2_weight_i = self.w2_weights[i]

                # Independent z sampling for actor training
                z_actor = torch.randn_like(action, device=device)

                # Transport map: π_i = T_s(state, z_actor)
                pi_i = actor_i(state, z_actor)

                # Critic freeze management:
                # - i==0: critic unfrozen (joint training)
                # - i>=1: critic frozen (one-time toggle at i==1)
                if i == 0:
                    # Ensure critic is unfrozen for i==0
                    for p in self.critic.parameters():
                        p.requires_grad_(True)
                elif i == 1:
                    # Freeze critic once at i==1, remains frozen for i>=2
                    for p in self.critic.parameters():
                        p.requires_grad_(False)

                Q_i = self.critic.Q1(state, pi_i)

                # W2 distance
                if i == 0:
                    # First actor: L2 to dataset actions
                    w2_i = ((pi_i - action) ** 2).mean()
                else:
                    # Subsequent actors: Sinkhorn distance to previous actor
                    ref_actor = self.actors[i - 1]
                    ref_actor.eval()
                    w2_i = per_state_sinkhorn(
                        actor_i,
                        ref_actor,
                        state,
                        K=4,
                        blur=0.05,
                        p=2,
                        use_ref_grad=False,
                        sinkhorn_loss=self._sinkhorn_loss,
                    )
                    ref_actor.train()

                # JKO loss: L = - lambda * E[Q] + w_i * W2
                denom = Q_i.abs().mean().detach().clamp_min(1e-6)
                actor_loss_i = -(Q_i.mean() / denom) + w2_weight_i * w2_i

                # Sequential backward: ensures critic freeze state is correct during backward
                opt = self.actor_optimizers[i]
                opt.zero_grad()
                actor_loss_i.backward()
                opt.step()

                # Log metrics for this actor
                metrics.update(
                    {
                        f"actor_{i}_loss": float(actor_loss_i.item()),
                        f"Q_{i}_mean": float(Q_i.mean().item()),
                        f"w2_{i}_distance": float(w2_i.item()),
                    }
                )

            # Unfreeze critic after all actor updates
            for p in self.critic.parameters():
                p.requires_grad_(True)

            # Soft-update targets
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            for actor, actor_target in zip(self.actors, self.actor_targets):
                for p, tp in zip(actor.parameters(), actor_target.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, filename: str):
        """Save critic and all actors."""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        for i in range(self.num_actors):
            torch.save(self.actors[i].state_dict(), filename + f"_actor_{i}")
            torch.save(
                self.actor_optimizers[i].state_dict(), filename + f"_actor_{i}_optimizer"
            )

    def load(self, filename: str):
        """Load critic and all actors."""
        map_location = device

        # Critic
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=map_location))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=map_location)
        )
        self.critic_target = copy.deepcopy(self.critic)

        # Actors (up to existing self.num_actors)
        for i in range(self.num_actors):
            self.actors[i].load_state_dict(
                torch.load(filename + f"_actor_{i}", map_location=map_location)
            )
            self.actor_optimizers[i].load_state_dict(
                torch.load(filename + f"_actor_{i}_optimizer", map_location=map_location)
            )
            self.actor_targets[i] = copy.deepcopy(self.actors[i])
