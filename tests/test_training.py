import numpy as np
import torch

from src.models.heads import MLPHead
from src.models.training import fit_head, split_target_hash


def test_fit_head_is_reproducible_and_reduces_l1_loss():
    features = torch.linspace(-1.0, 1.0, 24).reshape(-1, 1).repeat(1, 4)
    targets = features[:, 0] * 0.8 + 2.5

    torch.manual_seed(123)
    first = MLPHead(4, hidden_dim=8, second_hidden_dim=4)
    torch.manual_seed(123)
    second = MLPHead(4, hidden_dim=8, second_hidden_dim=4)
    first_history = fit_head(first, features, targets, epochs=25, learning_rate=0.01, batch_size=8, seed=11)
    second_history = fit_head(second, features, targets, epochs=25, learning_rate=0.01, batch_size=8, seed=11)

    assert first_history["loss"][0] > first_history["loss"][-1]
    assert first_history == second_history
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


def test_split_target_hash_is_order_invariant_but_value_sensitive():
    rows = [("b.wav", 2.0, 3.0), ("a.wav", 4.0, 5.0)]

    assert split_target_hash(rows) == split_target_hash(list(reversed(rows)))
    assert split_target_hash(rows) != split_target_hash([("b.wav", 2.1, 3.0), ("a.wav", 4.0, 5.0)])
