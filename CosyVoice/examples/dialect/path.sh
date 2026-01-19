#!/bin/bash
# Path configuration for dialect training

export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(dirname $(realpath $0))))"
