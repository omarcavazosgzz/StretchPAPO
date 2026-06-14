"""Compare local and remote directories via SSH.

Usage Examples:
    # Basic comparison (shows which files differ):
    python compare_toolkit_remote.py --local stretch_toolkit --remote hello-robot@192.168.137.215:/home/hello-robot/stretch_workspace/digital_twin/stretch_toolkit
    
    # With detailed diffs:
    python compare_toolkit_remote.py --local stretch_toolkit --remote hello-robot@192.168.137.215:/home/hello-robot/stretch_workspace/digital_twin/stretch_toolkit --diff
    
    # If SSH config is set up (so 'ssh hello-robot' works):
    python compare_toolkit_remote.py --local stretch_toolkit --remote hello-robot:/home/hello-robot/stretch_workspace/digital_twin/stretch_toolkit
"""
import os
import tempfile
import shutil
from pathlib import Path
import difflib
import click
import subprocess
import json


def is_remote_path(path):
    """Check if path is remote (format: user@host:/path or host:/path)."""
    return ':' in path and ('/' in path or '\\' in path)


def parse_remote_path(remote_path):
    """Parse remote path into (host, path) tuple."""
    if '@' in remote_path:
        # Format: user@host:/path
        host_part, path_part = remote_path.split(':', 1)
        return host_part, path_part
    else:
        # Format: host:/path
        host_part, path_part = remote_path.split(':', 1)
        return host_part, path_part


def download_remote_directory(host, remote_dir, temp_dir):
    """Download remote directory using rsync or scp."""
    click.secho(f"Downloading remote directory from {host}:{remote_dir}...", fg="white", dim=True)
    
    # Try rsync first (more efficient)
    rsync_cmd = [
        'rsync', '-az', '--exclude=__pycache__',
        f'{host}:{remote_dir}/', temp_dir
    ]
    
    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, check=True)
        click.secho("✓ Downloaded using rsync", fg="green", dim=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Rsync failed or not available, try scp with recursive copy
        click.secho("rsync not available, trying scp...", fg="yellow", dim=True)
        scp_cmd = ['scp', '-r', f'{host}:{remote_dir}', temp_dir]
        
        try:
            subprocess.run(scp_cmd, capture_output=True, text=True, check=True)
            click.secho("✓ Downloaded using scp", fg="green", dim=True)
            return True
        except subprocess.CalledProcessError as e:
            click.secho(f"Error downloading remote directory: {e.stderr}", fg="red")
            return False


def get_all_local_files(directory):
    """Get all files in a local directory recursively."""
    base_path = Path(directory)
    files = {}
    for file_path in base_path.rglob('*'):
        if file_path.is_file() and '__pycache__' not in str(file_path):
            rel_path = file_path.relative_to(base_path)
            files[str(rel_path).replace('\\', '/')] = file_path
    return files


def read_local_file_lines(file_path):
    """Read local file lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.readlines()
    except Exception as e:
        return [f"Error reading file: {e}\n"]


def compare_local_with_remote(local_dir, remote_path, show_diff=False):
    """Compare local directory with remote directory."""
    host, remote_dir = parse_remote_path(remote_path)
    
    click.secho(f"\nComparing:", fg="cyan", bold=True)
    click.secho(f"  Local:  {local_dir}", fg="cyan")
    click.secho(f"  Remote: {host}:{remote_dir}", fg="cyan")
    click.secho("=" * 80, fg="cyan")
    
    # Download remote directory to temp folder
    temp_dir = tempfile.mkdtemp(prefix='remote_toolkit_')
    try:
        if not download_remote_directory(host, remote_dir, temp_dir):
            return
        
        # Get the actual remote directory name from the path
        remote_dir_name = os.path.basename(remote_dir.rstrip('/'))
        downloaded_path = os.path.join(temp_dir, remote_dir_name)
        
        # If rsync was used, files are in temp_dir directly
        if not os.path.exists(downloaded_path):
            downloaded_path = temp_dir
        
        # Get local files
        click.secho("\nScanning local directory...", fg="white", dim=True)
        local_files = get_all_local_files(local_dir)
        
        # Get remote files (now local in temp)
        click.secho("Scanning downloaded remote directory...", fg="white", dim=True)
        remote_files = get_all_local_files(downloaded_path)
        
        all_files = set(local_files.keys()) | set(remote_files.keys())
        only_local = set(local_files.keys()) - set(remote_files.keys())
        only_remote = set(remote_files.keys()) - set(local_files.keys())
        common_files = set(local_files.keys()) & set(remote_files.keys())
        
        # Files only in local
        if only_local:
            click.secho(f"\n📁 Only in LOCAL:", fg="yellow", bold=True)
            for file in sorted(only_local):
                click.secho(f"  + {file}", fg="yellow")
        
        # Files only in remote
        if only_remote:
            click.secho(f"\n📁 Only in REMOTE:", fg="magenta", bold=True)
            for file in sorted(only_remote):
                click.secho(f"  - {file}", fg="magenta")
        
        # Compare common files
        different_files = []
        identical_files = []
        
        click.secho(f"\nComparing {len(common_files)} common files...", fg="white", dim=True)
        
        for file in sorted(common_files):
            local_path = local_files[file]
            remote_path = remote_files[file]
            
            local_lines = read_local_file_lines(local_path)
            remote_lines = read_local_file_lines(remote_path)
            
            if local_lines == remote_lines:
                identical_files.append(file)
            else:
                different_files.append((file, local_path, remote_path, local_lines, remote_lines))
    
    # Show identical files
        if identical_files:
            click.secho(f"\n✓ Identical files ({len(identical_files)}):", fg="green", bold=True)
            for file in identical_files:
                click.secho(f"  ✓ {file}", fg="green")
        
        # Show different files
        if different_files:
            click.secho(f"\n⚠ Different files ({len(different_files)}):", fg="red", bold=True)
            for file, local_path, remote_path, local_lines, remote_lines in different_files:
                click.secho(f"\n  ≠ {file}", fg="red", bold=True)
                
                if show_diff:
                    # Show unified diff
                    diff = difflib.unified_diff(
                        local_lines, remote_lines,
                        fromfile=f"LOCAL/{file}",
                        tofile=f"REMOTE/{file}",
                        lineterm=''
                    )
                    
                    diff_lines = list(diff)
                    if diff_lines:
                        click.secho("    " + "-" * 76, fg="blue")
                        for line in diff_lines[:100]:  # Limit output
                            line = line.rstrip()
                            if line.startswith('+++') or line.startswith('---'):
                                click.secho(f"    {line}", fg="blue", bold=True)
                            elif line.startswith('+'):
                                click.secho(f"    {line}", fg="green")
                            elif line.startswith('-'):
                                click.secho(f"    {line}", fg="red")
                            elif line.startswith('@@'):
                                click.secho(f"    {line}", fg="cyan")
                            else:
                                click.secho(f"    {line}", fg="white", dim=True)
                        
                        if len(diff_lines) > 100:
                            click.secho(f"    ... ({len(diff_lines) - 100} more lines)", fg="yellow")
                        click.secho("    " + "-" * 76, fg="blue")
        
        # Summary
        click.secho("\n" + "=" * 80, fg="cyan")
        click.secho("SUMMARY:", fg="cyan", bold=True)
        click.secho(f"  Total files in LOCAL: {len(local_files)}", fg="white")
        click.secho(f"  Total files in REMOTE: {len(remote_files)}", fg="white")
        click.secho(f"  Identical files: {len(identical_files)}", fg="green")
        click.secho(f"  Different files: {len(different_files)}", fg="red")
        click.secho(f"  Only in LOCAL: {len(only_local)}", fg="yellow")
        click.secho(f"  Only in REMOTE: {len(only_remote)}", fg="magenta")
        click.secho("=" * 80, fg="cyan")
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            click.secho(f"\n✓ Cleaned up temporary files", fg="green", dim=True)
        except Exception as e:
            click.secho(f"\n⚠ Could not clean up temp directory: {e}", fg="yellow", dim=True)

@click.command()
@click.option('--local', default='stretch_toolkit', help='Local directory to compare')
@click.option('--remote', required=True, help='Remote directory (format: user@host:/path or host:/path)')
@click.option('--diff', is_flag=True, help='Show detailed diff for different files')
def main(local, remote, diff):
    """Compare local directory with remote directory via SSH.
    
    Example:
        python compare_toolkit_remote.py --local stretch_toolkit --remote hello-robot@192.168.137.215:/home/hello-robot/stretch_workspace/digital_twin/stretch_toolkit
    """
    if not os.path.exists(local):
        click.secho(f"Error: Local directory '{local}' does not exist!", fg="red", bold=True)
        return
    
    if not is_remote_path(remote):
        click.secho(f"Error: '{remote}' doesn't look like a remote path (expected format: user@host:/path)", fg="red", bold=True)
        return
    
    compare_local_with_remote(local, remote, show_diff=diff)


if __name__ == '__main__':
    main()
