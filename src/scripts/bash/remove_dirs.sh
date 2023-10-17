#!/bin/bash

# Script to remove specified directories within the current working directory.
# It first lists all matching directories and asks for user confirmation before deletion.
#
# Usage:
# - To specify the folders to be listed and potentially removed in the current directory:
#   `./script.sh outputs lightning_logs __pycache__`
# - The script will list the directories and wait for a 'y/n' confirmation from the user.

remove_dirs() {
    dir_path="$PWD" # Default to current working directory
    files=("$@")    # Collect directory names from command-line arguments

    # List directories that will be removed
    echo "Directories to be removed in $dir_path:"
    for file in "${files[@]}"; do
        find "$dir_path" -type d -name "$file" -print
    done

    # Wait for user confirmation
    read -p "Are you sure you want to proceed? (y/n) " confirmation

    # Remove directories if user confirms
    if [ "$confirmation" == "y" ]; then
        for file in "${files[@]}"; do
            find "$dir_path" -type d -name "$file" -exec rm -r {} +
        done
        echo "Directories removed."
    else
        echo "Operation cancelled."
    fi
}

# All arguments are considered as directories to be removed
remove_dirs "$@"
