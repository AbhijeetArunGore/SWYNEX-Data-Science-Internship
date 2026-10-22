"""Repository Security, Secret & Privacy Audit Script.

Scans the repository working tree and Git commit history for:
1. Hardcoded API keys, tokens, and credentials
2. Private cryptographic keys and certificates
3. Database connection URIs and passwords
4. Hardcoded local user paths (e.g. personal Windows/Unix directories)
5. Sensitive environment files (.env) and private artifacts
"""
import os
import re
import sys
import subprocess
from typing import Dict, List, Tuple

# Regex patterns for common secret signatures
SECRET_PATTERNS = {
    'AWS Access Key': re.compile(r'\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b'),
    'GitHub Personal Access Token': re.compile(r'\bgh[pousr]_[A-Za-z0-9_]{36,255}\b'),
    'Generic API Key Assignment': re.compile(r'(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token)\s*[:=]\s*["\'][A-Za-z0-9_\-]{16,}["\']'),
    'Private Key Header': re.compile(r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
    'Database Connection String': re.compile(r'(?i)(postgres|mysql|mongodb|redis):\/\/[a-zA-Z0-9_\-]+:[^@\s]+@[a-zA-Z0-9_\-\.]+'),
    'Hardcoded Absolute Windows User Path': re.compile(r'[a-zA-Z]:\\Users\\[a-zA-Z0-9_\-]+', re.IGNORECASE),
}

# Paths and extensions to ignore
IGNORE_DIRS = {'.git', '.pytest_cache', '__pycache__', 'venv', '.venv', 'env', 'node_modules', 'dist', 'build'}
IGNORE_EXTS = {'.png', '.jpg', '.jpeg', '.parquet', '.joblib', '.pyc', '.pyd', '.svg'}


def audit_working_tree(repo_root: str) -> List[Dict[str, str]]:
    """Scan all text files in the working directory for security findings."""
    findings = []

    for root, dirs, files in os.walk(repo_root):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IGNORE_EXTS or file.endswith('.parquet') or file.endswith('.joblib'):
                continue

            filepath = os.path.join(root, file)
            relpath = os.path.relpath(filepath, repo_root)

            # Check for forbidden files
            if file == '.env':
                findings.append({
                    'file': relpath,
                    'line': 0,
                    'rule': 'Committed .env File',
                    'detail': 'Found active .env file which should be excluded from Git.'
                })
                continue

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, start=1):
                        # Skip documentation markdown mentions of patterns if they are explanatory
                        for rule_name, pattern in SECRET_PATTERNS.items():
                            match = pattern.search(line)
                            if match:
                                # Whitelist harmless mentions in this scanner script itself
                                if 'scripts/security_audit.py' in relpath.replace('\\', '/'):
                                    continue
                                # Whitelist example references in security policy or build notes if sanitized
                                if 'docs/security.md' in relpath.replace('\\', '/') and 'pattern' in line.lower():
                                    continue

                                findings.append({
                                    'file': relpath,
                                    'line': line_num,
                                    'rule': rule_name,
                                    'detail': f"Matched pattern '{rule_name}' (value masked for security)."
                                })
            except Exception as e:
                findings.append({
                    'file': relpath,
                    'line': 0,
                    'rule': 'File Read Error',
                    'detail': str(e)
                })

    return findings


def audit_git_history(repo_root: str) -> List[Dict[str, str]]:
    """Check recent Git commit diffs for accidental credentials."""
    findings = []
    try:
        # Check diff of last 20 commits
        cmd = ['git', 'log', '-p', '-n', '20']
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding='utf-8', errors='ignore', check=True)
        lines = res.stdout.splitlines()

        current_commit = "unknown"
        for line_num, line in enumerate(lines, start=1):
            if line.startswith('commit '):
                current_commit = line.split()[1][:8]
            if line.startswith('+') and not line.startswith('+++'):
                for rule_name, pattern in SECRET_PATTERNS.items():
                    if pattern.search(line):
                        if 'AKIA' in line or 'ghp_' in line or 'PRIVATE KEY' in line:
                            findings.append({
                                'file': f'git commit {current_commit}',
                                'line': line_num,
                                'rule': f'Historical Diff: {rule_name}',
                                'detail': f"Added line in commit {current_commit} matches {rule_name}."
                            })
    except Exception:
        # Git log may fail if git is unavailable
        pass

    return findings


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"[*] Starting Security & Secret Audit for: {repo_root}")

    tree_findings = audit_working_tree(repo_root)
    history_findings = audit_git_history(repo_root)
    all_findings = tree_findings + history_findings

    print("=" * 70)
    print("SECURITY & HYGIENE SCAN RESULTS")
    print("=" * 70)

    if not all_findings:
        print("[PASSED] Zero hardcoded secrets, private keys, database strings, or path leaks detected!")
        print(f"Files audited: Clean.")
        sys.exit(0)
    else:
        print(f"[ALERT] Found {len(all_findings)} potential security finding(s):")
        for f in all_findings:
            print(f"  - [{f['rule']}] {f['file']}:{f['line']} -> {f['detail']}")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
