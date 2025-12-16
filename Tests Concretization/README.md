# Tests Concretization Tool

A Python code generator that converts GraphWalker-style model and abstract test paths (JSON or txt format) into Python testing code.

### Input

- A model exported from GraphWalker.
- An abstract test path generated from the above model. The format could be JSON or plain txt.

### Output

Two Python files:

- Step functions file, containing vertex and edge functions and mapping tables.
- Path runner file that invokes the functions in the above file to enable future online testing.

### Usage

```
python py_tests_gen.py model.json -o [step functions Python file] --path [abstract path] --runner-out [tests runner Python file]
```
