# other libraries
import pytest

# own modules
from src.evaluate import main


@pytest.mark.order(7)
def test_accuracy() -> None:
    """
    This is the test for the accuracy in the test set.
    """

    # call evaluate
    accuracy_value: float = main("best_model")

    # check if accuracy is higher than 70%
    assert accuracy_value > 0.70, "Accuracy not higher than 70%"

    # check if accuracy is higher than 75%
    assert accuracy_value > 0.75, "Accuracy not higher than 75%"

    # check if accuracy is higher than 80%
    assert accuracy_value > 0.80, "Accuracy not higher than 80%"

    return None
