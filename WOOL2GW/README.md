# WOOL → GraphWalker Converter

A Python tool that converts WOOL dialogue scripts into GraphWalker model JSON. It also supports generate extra nodes/edges from an entrypoint context JSON.

### Inputs

A folder contains:

- WOOL file containing dialogue nodes.
- Optional context JSON files.

### Output

A GraphWalker JSON file containing one model with generated vertices, edges, actions and guards.

### Usage

```
python main.py [folder that contains WOOL files and context JSON files] --out [converted GraphWalker model]
```
