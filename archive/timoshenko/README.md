# Timoshenko Engine

Timoshenko Engine is developed in its own repository,
[Ayberkrk/timoshenko](https://github.com/Ayberkrk/timoshenko), and installed
as the `timoshenko-engine` package.

Cauren uses it only through `cauren_physics/timoshenko_adapter.py`, which
loads version 2.0 or newer when installed and otherwise falls back to Cauren's
own pure-Python implementations. Do not copy the package source into this
repository. Cauren-specific risk interpretation remains in Cauren.
