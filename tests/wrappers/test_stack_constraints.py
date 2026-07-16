from typing import ClassVar

import pytest

import envelope
from envelope.environment import StackConstraint, not_containing, not_inside
from envelope.wrappers import (
    PooledInitializationWrapper,
    VectorizingWrapper,
)
from envelope.wrappers.pooled_init_vmap_wrapper import PooledInitVmapWrapper
from envelope.wrappers.vmap_wrapper import VmapWrapper
from envelope.wrappers.wrapper import Wrapper
from tests.wrappers.helpers import ScalarToyEnv


class OuterTarget(Wrapper):
    pass


class OuterTargetSubclass(OuterTarget):
    pass


class OtherOuterTarget(Wrapper):
    pass


class InnerTarget(Wrapper):
    pass


class OtherInnerTarget(Wrapper):
    pass


class IntermediateWrapper(Wrapper):
    pass


class RejectOuterTargets(Wrapper):
    stack_constraints: ClassVar[tuple[StackConstraint, ...]] = (
        not_inside(OuterTarget, OtherOuterTarget),
    )


class RejectInnerTargets(Wrapper):
    stack_constraints: ClassVar[tuple[StackConstraint, ...]] = (
        not_containing(InnerTarget, OtherInnerTarget),
    )


class CustomVectorizer(Wrapper, VectorizingWrapper):
    pass


class RejectVectorizers(Wrapper):
    stack_constraints: ClassVar[tuple[StackConstraint, ...]] = (
        not_containing(VectorizingWrapper),
    )


class PoolUnsafeEnvironment(ScalarToyEnv):
    stack_constraints: ClassVar[tuple[StackConstraint, ...]] = (
        not_inside(PooledInitializationWrapper),
    )


def test_constraint_helpers_are_exposed_only_from_wrappers() -> None:
    from envelope.wrappers import StackConstraint as ExportedStackConstraint
    from envelope.wrappers import not_containing as exported_not_containing
    from envelope.wrappers import not_inside as exported_not_inside

    assert ExportedStackConstraint is StackConstraint
    assert exported_not_inside is not_inside
    assert exported_not_containing is not_containing
    assert not hasattr(envelope, "StackConstraint")
    assert not hasattr(envelope, "not_inside")
    assert not hasattr(envelope, "not_containing")


@pytest.mark.parametrize(
    "make_stack",
    [
        lambda: OuterTarget(RejectOuterTargets(ScalarToyEnv())),
        lambda: OuterTarget(IntermediateWrapper(RejectOuterTargets(ScalarToyEnv()))),
        lambda: OtherOuterTarget(RejectOuterTargets(ScalarToyEnv())),
    ],
    ids=["direct", "indirect", "multiple-match-types"],
)
def test_not_inside_searches_the_complete_outer_chain(make_stack) -> None:
    with pytest.raises(
        ValueError, match="RejectOuterTargets cannot be inside .*OuterTarget"
    ):
        make_stack()


@pytest.mark.parametrize(
    "make_stack",
    [
        lambda: RejectInnerTargets(InnerTarget(ScalarToyEnv())),
        lambda: RejectInnerTargets(IntermediateWrapper(InnerTarget(ScalarToyEnv()))),
        lambda: RejectInnerTargets(OtherInnerTarget(ScalarToyEnv())),
    ],
    ids=["direct", "indirect", "multiple-match-types"],
)
def test_not_containing_searches_the_complete_inner_chain(make_stack) -> None:
    with pytest.raises(
        ValueError, match="RejectInnerTargets cannot contain .*InnerTarget"
    ):
        make_stack()


def test_constraints_use_isinstance_for_subclasses() -> None:
    with pytest.raises(
        ValueError,
        match="RejectOuterTargets cannot be inside OuterTargetSubclass",
    ):
        OuterTargetSubclass(RejectOuterTargets(ScalarToyEnv()))


def test_unrelated_exact_types_do_not_conflict() -> None:
    stack = IntermediateWrapper(RejectOuterTargets(ScalarToyEnv()))
    assert isinstance(stack, IntermediateWrapper)


def test_marker_mixins_match_custom_wrapper_families() -> None:
    with pytest.raises(
        ValueError, match="RejectVectorizers cannot contain CustomVectorizer"
    ):
        RejectVectorizers(CustomVectorizer(ScalarToyEnv()))


def test_custom_base_environment_can_declare_constraints() -> None:
    with pytest.raises(
        ValueError,
        match="PoolUnsafeEnvironment cannot be inside PooledInitVmapWrapper",
    ):
        PooledInitVmapWrapper(
            PoolUnsafeEnvironment(),
            batch_size=2,
            pool_size=2,
        )


def test_builtin_vectorizer_markers_are_structural() -> None:
    vmap = VmapWrapper(ScalarToyEnv(), batch_size=2)
    pooled = PooledInitVmapWrapper(ScalarToyEnv(), batch_size=2, pool_size=2)

    assert isinstance(vmap, VectorizingWrapper)
    assert isinstance(pooled, VectorizingWrapper)
    assert isinstance(pooled, PooledInitializationWrapper)
