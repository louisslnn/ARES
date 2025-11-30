#!/usr/bin/env python3
"""
Simple script to view model results from results.csv
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.results import print_results_summary, get_results

if __name__ == "__main__":
    print_results_summary()

