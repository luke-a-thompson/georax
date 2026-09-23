from georax import post_lie_bracket
from georax._geometry.base import post_lie_bracket as implementation


def test_post_lie_bracket_is_exported() -> None:
    assert post_lie_bracket is implementation
