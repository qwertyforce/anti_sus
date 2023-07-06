#!/bin/bash

# Start the first process
screen -dm python anti_sus.py

# Start the second process
python reddit_stream.py