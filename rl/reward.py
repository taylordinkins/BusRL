"""Reward calculation for the Bus RL environment.

Implements the reward structure defined in the RL implementation plan:
- +1.0 per passenger delivered
- +0.1 bonus for stealing passengers from opponent lines
- +0.01 bonus for exclusive deliveries (destination not on opponent network)
- +0.1 for first train station connections
- -0.01 penalty for taking time stones
- Terminal: point differential vs first place
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from core.constants import Phase
from .config import RewardConfig, DEFAULT_REWARD_CONFIG, BoardConfig, DEFAULT_BOARD_CONFIG

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.board import NodeId


@dataclass
class StepRewardInfo:
    """Detailed breakdown of rewards for a single step.

    Useful for debugging and understanding agent behavior.
    """
    base_reward: float = 0.0
    delivery_reward: float = 0.0
    stolen_bonus: float = 0.0
    exclusive_bonus: float = 0.0
    station_connection_reward: float = 0.0
    time_stone_penalty: float = 0.0
    terminal_reward: float = 0.0

    @property
    def total(self) -> float:
        """Total reward for this step."""
        return (
            self.base_reward
            + self.delivery_reward
            + self.stolen_bonus
            + self.exclusive_bonus
            + self.station_connection_reward
            + self.time_stone_penalty
            + self.terminal_reward
        )


class RewardCalculator:
    """Calculates rewards for RL training.

    Tracks state across steps to detect events like:
    - Score changes (deliveries)
    - Time stone acquisitions
    - New train station connections
    """

    def __init__(
        self,
        config: RewardConfig = DEFAULT_REWARD_CONFIG,
        board_config: BoardConfig = DEFAULT_BOARD_CONFIG,
    ):
        """Initialize the reward calculator.

        Args:
            config: Reward configuration with reward values.
            board_config: Board configuration for station node IDs.
        """
        self.config = config
        self.board_config = board_config

        # Track which stations each player has connected to (persists across episode)
        self._stations_connected: dict[int, set[int]] = {}

    def reset(self) -> None:
        """Reset tracking state for a new episode."""
        self._stations_connected = {}

    def compute_reward(
        self,
        state: "GameState",
        prev_state: "GameState",
        player_id: int,
        done: bool,
        action_info: Optional[dict] = None,
    ) -> float:
        """Compute reward for a player after a step.

        Args:
            state: Current game state after the action.
            prev_state: Game state before the action.
            player_id: The player who took the action.
            done: Whether the game has ended.
            action_info: Optional action metadata (e.g., delivery details).

        Returns:
            Total reward for this step.
        """
        reward_info = self.compute_reward_detailed(
            state, prev_state, player_id, done, action_info
        )
        return reward_info.total

    def compute_reward_detailed(
        self,
        state: "GameState",
        prev_state: "GameState",
        player_id: int,
        done: bool,
        action_info: Optional[dict] = None,
    ) -> StepRewardInfo:
        """Compute detailed reward breakdown for a player.

        Args:
            state: Current game state after the action.
            prev_state: Game state before the action.
            player_id: The player who took the action.
            done: Whether the game has ended.
            action_info: Optional action metadata.

        Returns:
            StepRewardInfo with detailed reward breakdown.
        """
        info = StepRewardInfo()

        # Check for score changes (deliveries)
        info.delivery_reward = self._compute_delivery_reward(
            state, prev_state, player_id
        )

        # Check for delivery bonuses if action_info provided
        if action_info:
            stolen, exclusive = self._compute_delivery_bonuses(
                state, player_id, action_info
            )
            info.stolen_bonus = stolen
            info.exclusive_bonus = exclusive

        # Check for new train station connections
        info.station_connection_reward = self._check_station_connections(
            state, player_id
        )

        # Check for time stone penalties
        info.time_stone_penalty = self._compute_time_stone_penalty(
            state, prev_state, player_id
        )

        # Terminal reward
        if done:
            info.terminal_reward = self._compute_terminal_reward(state, player_id)

        return info

    def _compute_delivery_reward(
        self,
        state: "GameState",
        prev_state: "GameState",
        player_id: int,
    ) -> float:
        """Compute reward for score changes (passenger deliveries).

        Args:
            state: Current state.
            prev_state: Previous state.
            player_id: Player to compute reward for.

        Returns:
            Reward for score increase.
        """
        prev_score = prev_state.get_player(player_id).score
        curr_score = state.get_player(player_id).score
        score_delta = curr_score - prev_score

        if score_delta > 0:
            return score_delta * self.config.delivery_reward
        return 0.0

    def _compute_delivery_bonuses(
        self,
        state: "GameState",
        player_id: int,
        action_info: dict,
    ) -> tuple[float, float]:
        """Compute stolen and exclusive delivery bonuses.

        Args:
            state: Current state.
            player_id: Delivering player.
            action_info: Action metadata with delivery details.

        Returns:
            Tuple of (stolen_bonus, exclusive_bonus).
        """
        stolen_bonus = 0.0
        exclusive_bonus = 0.0

        # Check if this was a Vrroomm delivery
        if action_info.get("delivery"):
            delivery = action_info["delivery"]
            from_node = delivery.get("from_node")
            to_node = delivery.get("to_node")

            if from_node is not None and to_node is not None:
                # Stolen passenger bonus: passenger was on opponent's line
                if self._was_on_opponent_line(state, player_id, from_node):
                    stolen_bonus = self.config.stolen_passenger_bonus

                # Exclusive delivery bonus: destination not on opponent network
                if self._is_exclusive_destination(state, player_id, to_node):
                    exclusive_bonus = self.config.exclusive_delivery_bonus

        return stolen_bonus, exclusive_bonus

    def _was_on_opponent_line(
        self,
        state: "GameState",
        player_id: int,
        node_id: "NodeId",
    ) -> bool:
        """Check if a node was on any opponent's rail network.

        Args:
            state: Current state.
            player_id: Current player (not opponent).
            node_id: Node to check.

        Returns:
            True if any opponent has rails touching this node.
        """
        for player in state.players:
            if player.player_id == player_id:
                continue
            opponent_nodes = state.board.get_player_network_nodes(player.player_id)
            if node_id in opponent_nodes:
                return True
        return False

    def _is_exclusive_destination(
        self,
        state: "GameState",
        player_id: int,
        node_id: "NodeId",
    ) -> bool:
        """Check if destination node is not on any opponent's network.

        Args:
            state: Current state.
            player_id: Delivering player.
            node_id: Destination node.

        Returns:
            True if no opponent has rails touching this node.
        """
        return not self._was_on_opponent_line(state, player_id, node_id)

    def _check_station_connections(
        self,
        state: "GameState",
        player_id: int,
    ) -> float:
        """Check for new train station connections and award bonus.

        Args:
            state: Current state.
            player_id: Player to check.

        Returns:
            Reward for any new station connections.
        """
        reward = 0.0

        # Initialize tracking for this player if needed
        if player_id not in self._stations_connected:
            self._stations_connected[player_id] = set()

        # Get player's current network nodes
        player_nodes = state.board.get_player_network_nodes(player_id)

        # Check each train station
        for station_id in self.board_config.TRAIN_STATION_NODE_IDS:
            if station_id in player_nodes:
                if station_id not in self._stations_connected[player_id]:
                    # New connection!
                    self._stations_connected[player_id].add(station_id)
                    reward += self.config.station_connection_reward

        return reward

    def _compute_time_stone_penalty(
        self,
        state: "GameState",
        prev_state: "GameState",
        player_id: int,
    ) -> float:
        """Compute penalty for taking time stones.

        Args:
            state: Current state.
            prev_state: Previous state.
            player_id: Player to check.

        Returns:
            Penalty (negative value) for time stones taken.
        """
        prev_stones = prev_state.get_player(player_id).time_stones
        curr_stones = state.get_player(player_id).time_stones
        stones_delta = curr_stones - prev_stones

        if stones_delta > 0:
            return stones_delta * self.config.time_stone_penalty
        return 0.0

    def _compute_terminal_reward(
            self,
            state: "GameState",
            player_id: int,
        ) -> float:
        """Compute terminal reward based on point differential with draw handling.

        Rewards:
        • Win: (first - second) + won_game_bonus
        • Draw (tie for first): draw_bonus
        • Second place: (score - first) + second_place_bonus
        • Others: (score - first)

        This ensures:
        win > draw > second > others
        """

        # ---- Compute final scores ----
        final_scores = []
        for player in state.players:
            final_score = player.score - player.time_stones
            final_scores.append((final_score, player.player_id))

        # Sort descending by score
        final_scores.sort(reverse=True, key=lambda x: x[0])

        # Extract ordered scores
        scores_only = [s for s, _ in final_scores]

        first_place_score = scores_only[0]
        second_place_score = scores_only[1] if len(scores_only) > 1 else first_place_score

        # Identify all tied winners
        winners = {pid for score, pid in final_scores if score == first_place_score}

        # Player score
        player_score = next(score for score, pid in final_scores if pid == player_id)

        # Config bonuses
        win_bonus = getattr(self.config, "won_game_bonus", 0.0)
        draw_bonus = getattr(self.config, "draw_bonus", 0.0)
        second_bonus = getattr(self.config, "second_place_bonus", 0.0)

        # ---- Case 1: Draw (tie for first) ----
        if len(winners) > 1:
            if player_id in winners:
                return float(draw_bonus)
            else:
                # Non-winners still get shaped negative signal
                return float(player_score - first_place_score)

        # ---- Case 2: Unique winner ----
        winner_id = next(iter(winners))

        if player_id == winner_id:
            return float((first_place_score - second_place_score) + win_bonus)

        # ---- Case 3: Second place ----
        # Find true second place score (may be multiple)
        second_place_score = max(s for s in scores_only if s < first_place_score)
        second_place_players = {pid for score, pid in final_scores if score == second_place_score}

        if player_id in second_place_players:
            return float((player_score - first_place_score) + second_bonus)

        # ---- Case 4: Everyone else ----
        return float(player_score - first_place_score)

    def get_connected_stations(self, player_id: int) -> set[int]:
        """Get the set of station node IDs this player has connected to.

        Useful for debugging and visualization.

        Args:
            player_id: Player to query.

        Returns:
            Set of connected station node IDs.
        """
        return self._stations_connected.get(player_id, set()).copy()
