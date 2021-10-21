from typing import List, Union, Tuple, Dict, Type, Optional, Any

import math
import random
import numpy as np
import pandas as pd
import datetime
from dataclasses import dataclass
from obp.policy.base import BaseContextFreePolicy, BaseContextualPolicy


@dataclass
class BaseLinPolicy(BaseContextualPolicy):
    """Base class for contextual bandit policies using linear regression.

    Parameters
    ------------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    """

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.theta_hat = np.zeros((self.dim, self.n_actions))
        self.A_inv = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b = np.zeros((self.dim, self.n_actions))

        self.A_inv_temp = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b_temp = np.zeros((self.dim, self.n_actions))

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.

        Parameters
        ------------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        context: array-like, shape (1, dim_context)
            Observed context vector.

        """
        self.n_trial += 1
        self.action_counts[action] += 1
        # update the inverse matrix by the Woodbury formula
        self.A_inv_temp[action] -= (
            self.A_inv_temp[action]
            @ context.T
            @ context
            @ self.A_inv_temp[action]
            / (1 + context @ self.A_inv_temp[action] @ context.T)[0][0]
        )
        self.b_temp[:, action] += reward * context.flatten()
        if self.n_trial % self.batch_size == 0:
            self.A_inv, self.b = (
                np.copy(self.A_inv_temp),
                np.copy(self.b_temp),
            )


@dataclass
class LinUCB(BaseLinPolicy):
    """Linear Upper Confidence Bound.
    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.
    References
    --------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    epsilon: float = 0.0
    n_group: int = 0
    item_group: dict = {}
    group_count: dict = {}
    fairness_weight: list = []

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.policy_name = f"linear_ucb_{self.epsilon}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.
        Parameters
        ----------
        context: array
            Observed context vector.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        self.theta_hat = np.concatenate(
            [
                self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # dim * n_actions
        sigma_hat = np.concatenate(
            [
                np.sqrt(context @ self.A_inv[i] @ context.T)
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # 1 * n_actions
        ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()
        actions = ucb_scores.argsort()[::-1][: self.len_list]
        self.update_fairness_status(actions)

        return actions

    def update_fairness_status(self, action):
        for action in actions:
            self.group_count[self.item_group[action]] += 1

    def clear_group_count(self):
        self.group_count = {k: 0 for k in range(1, self.n_group + 1)}

    @property
    def propfair(self):
        propfair = 0
        total_exp = np.sum(list(self.group_count.values()))
        if total_exp > 0:
            propfair = np.sum(
                np.array(list(self.fairness_weight.values()))
                * np.log(1 + np.array(list(self.group_count.values())) / total_exp)
            )
        return propfair


@dataclass
class WFairLinUCB(LinUCB):
    """Linear Upper Confidence Bound.
    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.
    References
    --------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    epsilon: float = 0.0
    n_group: int = 0
    item_group: dict = {}
    group_count: dict = {}
    fairness_weight: list = []

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.policy_name = f"wfair_linear_ucb_{self.epsilon}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')"

        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.
        Parameters
        ----------
        context: array
            Observed context vector.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        self.theta_hat = np.concatenate(
            [
                self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # dim * n_actions
        sigma_hat = np.concatenate(
            [
                np.sqrt(context @ self.A_inv[i] @ context.T)
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # 1 * n_actions
        ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()

        wfair = (
            np.array(list(self.fairness_weight.values()))
            / np.sum(np.array(list(self.fairness_weight.values())))
        ) - (
            np.array(list(self.group_count.values()))
            / np.sum(list(self.group_count.values()))
        )
        ucb_scores = ucb_scores + (wfair * np.absolute(ucb_scores))

        actions = ucb_scores.argsort()[::-1][: self.len_list]
        self.update_fairness_status(actions)

        return actions


@dataclass
class FairLinUCB(LinUCB):
    """Linear Upper Confidence Bound.
    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.
    References
    --------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    epsilon: float = 0.0
    alpha: float = 0.0
    n_group: int = 0
    item_group: dict = {}
    group_count: dict = {}
    arm_count: dict = {}
    fairness_constraint: list = []

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.policy_name = f"fair_linear_ucb_{self.epsilon}_{self.alpha}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')"

        super().__post_init__()

    def calculate_score_fairness(self) -> np.array:
        fair = self.fairness_constraint * (
            np.sum(np.array(list(self.action_counts.value()))) - 1
        ) - np.array(list(self.action_counts.value()))

        return fair[~(fair < self.alpha)]

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.
        Parameters
        ----------
        context: array
            Observed context vector.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        A = self.calculate_score_fairness()
        if len(A) > 0:
            actions = [int(np.argmax(arm_scores))]
        else:
            self.theta_hat = np.concatenate(
                [
                    self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                    for i in np.arange(self.n_actions)
                ],
                axis=1,
            )  # dim * n_actions
            sigma_hat = np.concatenate(
                [
                    np.sqrt(context @ self.A_inv[i] @ context.T)
                    for i in np.arange(self.n_actions)
                ],
                axis=1,
            )  # 1 * n_actions
            ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()
            actions = ucb_scores.argsort()[::-1][: self.len_list]

        self.update_fairness_status(actions)
        return actions

    def update_fairness_status(self, action):
        for action in actions:
            self.group_count[self.item_group[action]] += 1
            self.arm_count[action] += 1

    def clear_group_count(self):
        self.group_count = {k: 0 for k in range(1, self.n_group + 1)}
        self.arm_count = {k: 0 for k in range(self.n_actions)}
