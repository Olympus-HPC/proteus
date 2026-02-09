# Checks that public headers do not include internal headers.

import os
import re
import sys
from pathlib import Path

def check_public_headers(include_dir):
    public_headers = Path(include_dir).rglob('*.h')
    found_error = False

    for header in public_headers:
        with open(header) as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments
                if line.strip().startswith('//'):
                    continue
                if re.search(r'#include\s+[<"].*proteus/impl.*[>"]', line):
                    print(f"::error file={header},line={line_num}::Public header includes internal header: {line.strip()}")
                    found_issue = True
    if found_error:
       sys.exit(1)
    else:
        print("All public headers are clean.")

if __name__ == '__main__':
    check_public_headers('include/proteus')
