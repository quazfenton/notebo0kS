#!/usr/bin/env python
import re

with open('hypothesisPredictorTemp.py', 'r') as f:
    content = f.read()

# Fix __init__ methods
content = re.sub(r'def init\(', 'def __init__(', content)

# Fix common indentation patterns for class bodies
lines = content.split('\n')
fixed = []
in_class = False
class_indent = 0

for line in lines:
    stripped = line.strip()
    current_indent = len(line) - len(line.lstrip()) if line.strip() else 0
    
    # Detect class definition
    if stripped.startswith('class ') and ':' in stripped:
        in_class = True
        class_indent = current_indent
        fixed.append(line)
        continue
    
    # Detect method definition
    if in_class and stripped.startswith('def ') and ':' in stripped:
        # Ensure method is indented relative to class
        if current_indent <= class_indent:
            fixed.append(' ' * (class_indent + 4) + stripped)
        else:
            fixed.append(line)
        continue
    
    # Fix docstrings in classes
    if in_class and (stripped.startswith('"""') or stripped.startswith("'''")):
        if current_indent <= class_indent:
            fixed.append(' ' * (class_indent + 4) + stripped)
        else:
            fixed.append(line)
        continue
    
    # Fix class body content (not methods, not comments, not empty)
    if in_class and stripped and not stripped.startswith('#') and not stripped.startswith('def ') and not stripped.startswith('class '):
        if current_indent <= class_indent:
            fixed.append(' ' * (class_indent + 4) + stripped)
        else:
            fixed.append(line)
        continue
    
    # Reset class tracking when we hit a top-level statement
    if current_indent == 0 and stripped and not stripped.startswith('#'):
        in_class = False
    
    fixed.append(line)

with open('hypothesisPredictorTemp.py', 'w') as f:
    f.write('\n'.join(fixed))

print('Systematic indentation fix applied')
