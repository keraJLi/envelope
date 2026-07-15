# Spaces

## Space Semantics

Envelope has two basic spaces: `Discrete` and `Continuous`. Both of them may have any
shape, where `Discrete` naturally expresses a multi-discrete space if it's shape is
any bigger than `()`. For convenience, both basic spaces have a `from_shape` method that
creates a space with a full array of the given shape.

The two basic spaces can be nested in any PyTree, composing them into a `PyTreeSpace`.
The `PyTreeSpace`s methods and properties are mapped onto the PyTree of subspaces,
treating them as the leaves. For example

```python
space = PyTreeSpace({
  "foo": Discrete([3, 5]),
  "bar": Continuous(low=-1.0, high=1.0)
})
space.shape  # {'foo': (2,), 'bar': ()}
space.dtype  # {'foo': dtype('int32'), 'bar': dtype('float32')}
```

Spaces can be batched using `BatchedSpace`, which returns a view that prepends a batch
dimension and vectorizes `sample` and `contains`.

`contains` is strict about pytree structure and every event/batch shape. Discrete
candidates may use any non-boolean integer width. Continuous candidates may use any
non-boolean real numeric dtype; their values still have to satisfy the declared bounds.
Shape, structure, dtype-category, or bounds mismatches return a scalar JAX false value.

Space constructors reject invalid concrete definitions eagerly. Continuous sampling
supports finite, one-sided, and fully unbounded dimensions without producing NaNs.

The constraints on the `PyTreeSpace`'s members imply a strict ordering on the
construction of spaces:
```
Discrete / Continuous → PyTreeSpace → BatchedSpace
```

## API Reference

::: envelope.spaces.Space

::: envelope.spaces.Discrete

::: envelope.spaces.Continuous

::: envelope.spaces.PyTreeSpace

::: envelope.spaces.BatchedSpace
