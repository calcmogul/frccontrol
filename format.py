#!/usr/bin/env python3

import subprocess
import sys

subprocess.run([sys.executable, "-m", "black", "-q", "."])
