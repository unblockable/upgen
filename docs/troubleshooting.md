# Troubleshooting

This file contains the issues and their corrections/explanations.

---

1. `python generate.py --help` command works when the current directory is src/ but from upgen/ directory 'python src/generate.py --help' does not work

Explanation: The generate.py file has `sys.path.insert(0, './greeting')` on line 17.
This assumes that the greeting directory is present in the current directory.
This is true for src/ directory. However, from upgen/ directory, it isn't true, so the path doesn't work and we get an error.
